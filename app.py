"""Streamlit UI with persistent, conversation-aware chat history."""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

# Streamlit Community Cloud stores deployed credentials in st.secrets, while
# src.rag reads configuration through os.getenv(). Bridge only the known flat
# configuration keys before importing src.rag. Local .env remains supported.
_STREAMLIT_SECRET_KEYS = (
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_CHAT_DEPLOYMENT",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "SARVAM_API_KEY",
    "SARVAM_CHAT_MODEL",
    "SARVAM_MAX_TOKENS",
    "SARVAM_REASONING_EFFORT",
    "SARVAM_CHAT_URL",
    "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME",
    "PINECONE_NAMESPACE",
    "PINECONE_CLOUD",
    "PINECONE_REGION",
    "DATA_DIR",
    "TOP_K",
    "RETRIEVAL_CANDIDATES",
)


def load_streamlit_secrets_into_environment() -> None:
    try:
        for key in _STREAMLIT_SECRET_KEYS:
            value = st.secrets.get(key)
            if value is not None and str(value).strip():
                # Streamlit secrets take precedence over a local .env value.
                os.environ[key] = str(value)
    except Exception:
        # No secrets.toml is expected during ordinary local development.
        pass


load_streamlit_secrets_into_environment()

from src.rag import answer_question, build_index, index_stats, run_dev_eval

DB_PATH = ROOT / "storage" / "chat_history.sqlite3"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def db_connection():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def initialize_chat_db():
    with db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                content TEXT NOT NULL,
                sources_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id)")


def create_conversation(title="New conversation"):
    conversation_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with db_connection() as conn:
        conn.execute(
            "INSERT INTO conversations(id,title,created_at,updated_at) VALUES(?,?,?,?)",
            (conversation_id, title, now, now),
        )
    return conversation_id


def list_conversations():
    with db_connection() as conn:
        return conn.execute(
            "SELECT id,title,created_at,updated_at FROM conversations ORDER BY updated_at DESC"
        ).fetchall()


def get_messages(conversation_id):
    with db_connection() as conn:
        rows = conn.execute(
            "SELECT role,content,sources_json FROM messages WHERE conversation_id=? ORDER BY id",
            (conversation_id,),
        ).fetchall()
    result = []
    for row in rows:
        try:
            sources = json.loads(row["sources_json"] or "[]")
        except json.JSONDecodeError:
            sources = []
        result.append({"role": row["role"], "content": row["content"], "sources": sources})
    return result


def save_message(conversation_id, role, content, sources=None):
    now = datetime.now(timezone.utc).isoformat()
    with db_connection() as conn:
        conn.execute(
            "INSERT INTO messages(conversation_id,role,content,sources_json,created_at) VALUES(?,?,?,?,?)",
            (conversation_id, role, content, json.dumps(sources or [], ensure_ascii=False), now),
        )
        if role == "user":
            row = conn.execute(
                "SELECT title FROM conversations WHERE id=?", (conversation_id,)
            ).fetchone()
            if row and row["title"] == "New conversation":
                title = content.strip().replace("\n", " ")
                title = title[:58] + ("…" if len(title) > 58 else "")
                conn.execute("UPDATE conversations SET title=? WHERE id=?", (title or "New conversation", conversation_id))
        conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, conversation_id))


def delete_conversation(conversation_id):
    with db_connection() as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))


def render_sources(sources, expanded=False):
    if not sources:
        return
    with st.expander("📚 Sources and retrieved evidence", expanded=expanded):
        for source in sources:
            location_type = source.get("location_type", "page")
            location = source.get("location", "")
            st.markdown(f"**{source.get('file', 'Source')} — {location_type} {location}**")
            if source.get("similarity") is not None:
                st.caption(f"Retrieval similarity: {source['similarity']}")
            if source.get("text"):
                st.write(source["text"])


st.set_page_config(
    page_title="CivicDataLab | Gender Budget AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
.block-container {padding-top:1.5rem;padding-bottom:2rem;max-width:1250px;}
[data-testid="stChatMessage"] {border:1px solid rgba(128,128,128,.16);border-radius:14px;padding:1rem;}
.small-muted {font-size:.86rem;opacity:.72;}
</style>
""", unsafe_allow_html=True)

initialize_chat_db()
if "active_conversation_id" not in st.session_state:
    existing = list_conversations()
    st.session_state.active_conversation_id = existing[0]["id"] if existing else create_conversation()

st.title("📊 Indian Gender Budget Research Assistant")
st.caption("A multilingual research chatbot over Union and state gender-budget documents, with conversation-aware chat history.")

# Sidebar: conversations and system controls
with st.sidebar:
    st.header("💬 Conversations")
    if st.button("＋ New conversation", use_container_width=True):
        st.session_state.active_conversation_id = create_conversation()
        st.rerun()

    conversations = list_conversations()
    ids = [row["id"] for row in conversations]
    active_id = st.session_state.active_conversation_id
    if active_id not in ids:
        active_id = create_conversation()
        st.session_state.active_conversation_id = active_id
        conversations = list_conversations()
        ids = [row["id"] for row in conversations]

    titles = {row["id"]: f"{row['title']}  ·  {row['updated_at'][:10]}" for row in conversations}
    selected = st.selectbox(
        "Open saved conversation",
        options=ids,
        index=ids.index(active_id),
        format_func=lambda value: titles.get(value, value),
        key="conversation_picker",
    )
    if selected != st.session_state.active_conversation_id:
        st.session_state.active_conversation_id = selected
        st.rerun()

    left, right = st.columns(2)
    with left:
        if st.button("Clear current", use_container_width=True):
            delete_conversation(st.session_state.active_conversation_id)
            st.session_state.active_conversation_id = create_conversation()
            st.rerun()
    with right:
        if st.button("Delete current", use_container_width=True):
            delete_conversation(st.session_state.active_conversation_id)
            rows = list_conversations()
            st.session_state.active_conversation_id = rows[0]["id"] if rows else create_conversation()
            st.rerun()

    st.caption("Chat history is stored in local SQLite (`storage/chat_history.sqlite3`). On Streamlit Community Cloud, local files are not guaranteed to persist across restarts or redeployments. This demo also uses one shared conversation database, so keep the deployed app private until per-user data isolation and a managed database are added.")

    st.divider()
    st.header("Workspace")
    st.markdown("**Model routing**")
    st.markdown("- 🇬🇧 English: Azure OpenAI")
    st.markdown("- 🇮🇳 Hindi / Hinglish: Sarvam-105B")
    st.markdown("- 🔎 Embeddings + retrieval: Azure OpenAI + Pinecone")
    st.divider()
    st.subheader("Connection status")
    checks = [
        ("Azure OpenAI", ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_CHAT_DEPLOYMENT", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]),
        ("Sarvam", ["SARVAM_API_KEY"]),
        ("Pinecone", ["PINECONE_API_KEY"]),
    ]
    for label, keys in checks:
        ready = all(os.getenv(key, "").strip() for key in keys)
        st.write(("🟢 " if ready else "🔴 ") + label + (" configured" if ready else " missing settings"))

    st.divider()
    st.subheader("Knowledge base")
    corpus_files = sorted(path for path in (ROOT / "data" / "corpus").glob("*") if path.is_file())
    st.metric("Source files", len(corpus_files))
    with st.expander("View corpus files"):
        for path in corpus_files:
            st.caption(path.name)

    if st.button("🔄 Build / rebuild Pinecone index", use_container_width=True):
        try:
            with st.status("Extracting documents, generating embeddings and uploading to Pinecone...", expanded=True) as status:
                count = build_index(reset=True)
                status.update(label=f"Pinecone index ready — {count} chunks indexed", state="complete")
            st.success(f"Indexed {count} text chunks.")
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")
    try:
        stats = index_stats()
        st.metric("Indexed vectors", stats["namespace_vector_count"])
        st.caption(f"Vector dimension: {stats['dimension']}")
    except Exception as exc:
        st.caption(f"Index stats unavailable: {exc}")
    st.divider()
    st.caption("Credentials are read from your local `.env`. Never commit that file.")

chat_tab, eval_tab, architecture_tab = st.tabs(["💬 Chatbot", "🧪 Evaluation", "⚙️ How it works"])

with chat_tab:
    st.subheader("Ask a question")
    st.markdown("Try a follow-up such as **“What about the previous financial year?”** or **“Compare that with Bihar.”**")
    conversation_id = st.session_state.active_conversation_id
    messages = get_messages(conversation_id)

    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_sources(message.get("sources", []))

    prompt = st.chat_input("Ask about a scheme, state, financial year, or ask a follow-up…")
    if prompt:
        # Snapshot prior turns before saving the new question. The resolver uses these turns
        # to expand elliptical follow-ups into a standalone question for retrieval.
        prior_history = [{"role": m["role"], "content": m["content"]} for m in messages[-10:]]
        save_message(conversation_id, "user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Resolving context, searching Pinecone and preparing a grounded answer..."):
                    result = answer_question(prompt, history=prior_history)
                if result["answered"]:
                    st.markdown(result["answer"])
                else:
                    st.warning("I couldn't find enough reliable evidence in the indexed documents. Try specifying the state, scheme and financial year.")
                st.caption(f"Answer model: {result.get('model')} · Vector DB: Pinecone")
                if result.get("resolved_question") and result["resolved_question"].strip() != prompt.strip():
                    st.caption(f"Understood as: {result['resolved_question']}")
                for warning in result.get("warnings", []):
                    st.warning(warning)
                if result.get("diagnostic") and not result.get("answered"):
                    st.caption(f"Why the assistant abstained: {result['diagnostic']}")
                if result.get("retrieval_query") and result["retrieval_query"] != prompt:
                    st.caption(f"Search query: {result['retrieval_query']}")
                render_sources(result.get("evidence", []), expanded=True)
                answer_text = result["answer"] if result["answered"] else "I couldn't determine this from the indexed documents."
                save_message(conversation_id, "assistant", answer_text, result.get("evidence", []))
            except Exception as exc:
                error = f"Request failed: {exc}"
                st.error(error)
                save_message(conversation_id, "assistant", error)
        st.rerun()

with eval_tab:
    st.subheader("Evaluation and regression testing")
    st.caption("Use the development set to check retrieval/source matching before running the 18 hidden-answer evaluation questions.")

    dev_file = ROOT / "data" / "questions_dev.jsonl"
    dev_questions = [json.loads(line) for line in dev_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    st.write(f"**{len(dev_questions)} development questions** include gold answers and expected source files.")

    if st.button("🧪 Run 12 development regression questions", type="secondary"):
        with st.spinner("Running development questions and comparing returned source files..."):
            try:
                dev_report = run_dev_eval()
                dev_output = ROOT / "dev_answers.jsonl"
                st.success(
                    f"Development run complete: {dev_report['answered']} answered; "
                    f"{dev_report['expected_source_file_matches']} expected source-file matches "
                    f"out of {dev_report['questions']}."
                )
                st.download_button(
                    "⬇ Download dev_answers.jsonl",
                    data=dev_output.read_bytes(),
                    file_name="dev_answers.jsonl",
                    mime="application/jsonl",
                )
                dev_rows = [json.loads(line) for line in dev_output.read_text(encoding="utf-8").splitlines() if line.strip()]
                st.dataframe(
                    [{
                        "ID": row["question_id"],
                        "Answered": row["answered"],
                        "Expected source": ", ".join(row["expected_source_files"]),
                        "Actual source": ", ".join(sorted({s.get("file", "") for s in row["sources"]})),
                        "Source match": row["source_file_match"],
                        "Answer": row["answer"],
                    } for row in dev_rows],
                    use_container_width=True,
                    hide_index=True,
                )
            except Exception as exc:
                st.error(f"Development evaluation failed: {exc}")

    st.divider()
    eval_file = ROOT / "data" / "questions_eval.jsonl"
    questions = [json.loads(line) for line in eval_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    st.write(f"**{len(questions)} evaluation questions** are included in the assignment.")

    if st.button("▶ Run all 18 evaluation questions", type="primary"):
        rows = []
        progress = st.progress(0)
        status = st.empty()
        output_path = ROOT / "answers.jsonl"
        error_count = 0
        with output_path.open("w", encoding="utf-8") as out:
            for i, question in enumerate(questions, 1):
                status.write(f"Processing {question['question_id']} ({i}/{len(questions)})")
                try:
                    result = answer_question(question["question"], question.get("lang"))
                except Exception as exc:
                    error_count += 1
                    st.error(f"{question['question_id']}: {type(exc).__name__}: {exc}")
                    result = {"answered": False, "answer": "", "sources": [], "diagnostic": str(exc)}
                record = {
                    "question_id": question["question_id"],
                    "answered": bool(result.get("answered")),
                    "answer": result.get("answer", "") if result.get("answered") else "",
                    "sources": result.get("sources", []) if result.get("answered") else [],
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                rows.append({
                    "ID": question["question_id"],
                    "Language": question.get("lang", ""),
                    "Question": question["question"],
                    "Answered": record["answered"],
                    "Answer": record["answer"],
                    "Sources": str(record["sources"]),
                    "Diagnostic": result.get("diagnostic", ""),
                })
                progress.progress(i / len(questions))
        answered_count = sum(1 for row in rows if row["Answered"])
        status.success(
            f"Evaluation finished: {answered_count}/{len(questions)} answered; "
            f"{len(questions)-answered_count} abstained; {error_count} execution errors."
        )
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇ Download answers.jsonl",
            data=output_path.read_bytes(),
            file_name="answers.jsonl",
            mime="application/jsonl",
        )

    if (ROOT / "answers.jsonl").exists():
        st.download_button(
            "⬇ Download latest answers.jsonl",
            data=(ROOT / "answers.jsonl").read_bytes(),
            file_name="answers.jsonl",
            mime="application/jsonl",
        )

with architecture_tab:
    st.subheader("Conversation-aware RAG flow")
    st.markdown("""
    1. **Conversation memory:** Recent user and assistant turns are loaded from SQLite for the selected conversation.
    2. **Follow-up resolution:** The appropriate chat model rewrites contextual questions into standalone questions, preserving schemes, state, year and units.
    3. **Language routing:** English uses Azure OpenAI; Hindi/Hinglish uses Sarvam-105B.
    4. **Metadata filtering:** State and (for Delhi) financial-year routing restrict retrieval to the appropriate source document.
    5. **Hybrid retrieval:** Pinecone dense matches are fused with local keyword ranking to improve scheme-name and table-row retrieval.
    6. **Grounded answer:** The model receives retrieved passages and must abstain when the jurisdiction, year, scheme or requested metric is unsupported.
    7. **Financial validation:** Explicit thousand/lakh-to-crore conversions are checked against deterministic formulas.
    8. **Local chat archive:** Questions, answers and source evidence are stored in SQLite. On Community Cloud, local file persistence is not guaranteed; a managed database is needed for durable multi-user history.
    """)
    st.info("Conversation history is stored in this app's local SQLite database. Back it up before moving the project. Streamlit Community Cloud does not guarantee durable local-file storage; use a managed database for long-term or multi-user history.")
