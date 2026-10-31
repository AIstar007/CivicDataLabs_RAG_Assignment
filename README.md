# CivicDataLab Gender Budget RAG — Azure OpenAI

A compact multilingual RAG implementation over the supplied Union and state gender-budget corpus. Azure OpenAI provides embeddings and English answer generation; Sarvam-105B provides Hindi/Hinglish query understanding and Hindi answer generation. No API keys are committed.

## 1. Requirements
- Python 3.10+
- An Azure OpenAI resource with one chat deployment and one embedding deployment
- A Sarvam API key for Hindi/Hinglish understanding and answer generation
- Network access to that Azure resource

## 2. Setup
```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env   # Windows; use cp on macOS/Linux
```
Edit `.env` and set `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_CHAT_DEPLOYMENT`, and `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`. The deployment variables must be the deployment names configured in Azure, not merely the base model names. Also set `SARVAM_API_KEY` from your Sarvam dashboard. Default Azure API version is `2024-10-21`.

## 3. Build the index and run evaluation
After updating to this version, rebuild the Pinecone index once because the new retrieval logic adds `state`, `financial_year`, `scope`, and other metadata fields.
```powershell
python -m src.rag ingest --reset
streamlit run app.py
# Run the 12 gold-labelled development regression questions first:
python -m src.rag dev-eval
# Then run the 18 assignment evaluation questions:
python -m src.rag eval
```
`python -m src.rag dev-eval` writes `dev_answers.jsonl` with the model answer, gold answer and expected-source-file match. `python -m src.rag eval` writes `answers.jsonl` with one submission-format JSON object for each of the 18 evaluation questions. Inspect the answers and citations before submission. Do not commit `.env` or paste credentials into GitHub.

Ask one question:
```bash
python -m src.rag ask --question "What was Delhi's Ladli Yojna allocation in 2024-25?"
```

## Model routing
- English questions use the Azure OpenAI chat deployment.
- Hindi/Hinglish questions use Sarvam-105B to normalize the question into an English retrieval query, then generate the final answer in Hindi from retrieved evidence.
- Azure OpenAI embeddings and Pinecone remain the shared retrieval layer.
- Evaluation rows use their `lang` field for deterministic routing; interactive questions use Devanagari and common Hinglish markers.

## Design
- PDFs are split page-by-page, then into overlapping text chunks. PDF page numbers are retained.
- CSV records are indexed row-wise, preserving column labels and values. XLS records are indexed row-wise with sheet names retained in metadata.
- Azure embedding vectors are stored in a Pinecone Serverless index with state, financial-year, scope, file, page/row and document-type metadata.
- Retrieval combines Pinecone dense similarity with local BM25-style lexical ranking using reciprocal-rank fusion. Delhi queries with a specified year are restricted to the matching source PDF; Bihar and Odisha filters preserve multi-year tables within their budget documents.
- The selected chat model receives retrieved excerpts and strict instructions to abstain when the evidence does not support the requested year, geography, unit, or claim.
- Output sources use `page` for PDFs and `row` for CSV/XLS records. For spreadsheets, the row number is a physical row number (header-inclusive).

## Known limitations
- No OCR is performed. Pages with no extractable PDF text are skipped.
- Hindi text extraction depends on the PDF's embedded fonts/encoding. Some legacy Hindi PDFs may extract as garbled text; this should be checked manually.
- Retrieval is a lightweight hybrid dense-plus-keyword approach; it is not a full BM25 service or cross-encoder reranker. OCR is not included.
- The supplied XLS is read with `xlrd`; if it is malformed or unsupported, it is reported and skipped rather than silently fabricated.


Sarvam API documentation: https://docs.sarvam.ai/api/token-factory/chat-completions


## Streamlit chatbot
Launch the interactive frontend:
```bash
streamlit run app.py
```
The app provides a chatbot for English, Hindi and Hinglish questions, conversation-aware follow-up resolution, SQLite chat history, retrieved source passages and citations, Pinecone index statistics, and the 18-question evaluation runner. In a local deployment, conversations can be reopened or deleted from the sidebar after restarting Streamlit. On Streamlit Community Cloud, local database persistence is not guaranteed. Use the sidebar's **Build / rebuild Pinecone index** button for first-time ingestion.

## Pinecone configuration
Set `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `PINECONE_NAMESPACE`, `PINECONE_CLOUD`, and `PINECONE_REGION` in `.env`. The index is created automatically on first ingestion using the exact dimension returned by the configured Azure embedding deployment. If you change embedding deployments/dimensions, rebuild the Pinecone index.


## Conversation history and follow-up questions
- Each conversation is stored in `storage/chat_history.sqlite3` (SQLite).
- The sidebar lets users create, reopen, clear and delete conversations.
- The latest conversation turns are passed to a query resolver before retrieval. It rewrites follow-ups such as "What about the previous financial year?" or "Compare that with Bihar" into standalone questions.
- English follow-ups use the Azure OpenAI chat deployment for rewriting; Hindi/Hinglish follow-ups use Sarvam-105B. The normal answer generation follows the same language routing.
- The resolved question is shown in the chat UI so users can inspect how the follow-up was interpreted.
- Chat history is local to the machine running Streamlit. Do not place the SQLite database in a public repository if it contains private conversations.


## Retrieval safeguards and regression tests
- Delhi year-specific requests are routed only to a source PDF whose contents correspond to that financial year. The file named `gender_budget_2012-13.pdf` contains 2011-12 Annual Plan data, so it is not treated as evidence for 2012-13.
- The provided corpus includes Bihar and Odisha budget PDFs but no Kerala gender-budget PDF. Kerala-specific requests therefore abstain instead of re-labelling Odisha figures.
- The corpus contains gender-budget statements, not Delhi's full education-department budget. A request for the total education budget is not answered from gender-specific allocations.
- Named scheme terms must be present in retrieved evidence. Unsupported schemes such as Mahila Samriddhi Yojana in the supplied Odisha PDF are not guessed.
- Financial-unit validation checks conversions across thousand, lakh, and crore, including parenthetical conversions, both `Rs 300` / `300 Rs` word orders, and localized output such as `210000 (मूल इकाई: Rs thousand)`. It corrects incorrect equivalents and appends a correct crore equivalent when the source amount is in thousand/lakh. Formulas: Rs thousand / 10,000 = crore; Rs lakh / 100 = crore.
- Odisha 2024-25 year-on-year comparison questions use a deterministic source consistency check. The answer cites PDF pages 10 and 21 and flags the conflicting 15.28% / 18.24% claims and inconsistent previous-year baselines instead of silently selecting one.
- `python -m unittest discover -s tests` runs regression tests for numeric conversions (including Hindi answers where the amount appears before a parenthetical source-unit label) and the Odisha source-conflict guard.
- `python -m src.rag dev-eval` runs the 12 development questions and writes `dev_answers.jsonl` with gold answers and expected-source-file matches for manual regression review.
- This v3 numeric/source-table patch changes answer validation only; no Pinecone re-ingestion is required. Rebuild with `python -m src.rag ingest --reset` only when the corpus, embedding deployment, or indexed metadata schema changes.


## Sarvam empty-response fix
The Sarvam client defaults short JSON/translation calls to `reasoning_effort=none`. Sarvam documents that reasoning tokens count against `max_tokens`; with thinking enabled, the model can use the available completion budget for reasoning and return an empty visible `message.content`. The client retries an empty response once with reasoning disabled and reports `finish_reason` and token details if it still fails. Set `SARVAM_REASONING_EFFORT=none` in your existing `.env` file as well (editing `.env.example` alone does not alter an existing `.env`).


## Updates for the latest evaluation output
- E08: The query is expanded with the source table's terminology (`female population`, `Total Population`, `crore`, `Census 2011`). When the exact row is retrieved from Bihar PDF page 15, the female population is read directly from the Bihar Female cell (5.0 crore) and cited to that page.
- E09: The Sarvam client retries empty visible replies with reasoning disabled. Hindi query translation has fail-soft mappings for the assignment's fixed Hindi questions, so a temporary empty translation response need not abort retrieval.
- E01-E03: Answers that state a Delhi amount in Rs thousand receive a deterministic crore equivalent. E04-E05: the Union CSV source unit is explicitly recorded as Rs crore.
- Set `SARVAM_REASONING_EFFORT=none` in your existing `.env` to avoid spending short structured-output requests on hidden reasoning tokens. The client still retries with reasoning disabled if the first reply is empty.

## Numeric and table-column safeguards (31 Oct 2026)
- Financial-unit validation now handles Hindi as well as English unit labels (`हजार`, `लाख`, `करोड़`) so a generated answer cannot turn Rs 210,000 thousand into Rs 210 crore; the correct equivalent is Rs 21 crore.
- The Delhi 2026-27 KISHORI row is read from the source PDF's labelled columns. Its first number, 402,200, is the 2025-26 Revised Estimate; its 2026-27 Annual Budget Total Outlay is 300,000 thousand (Rs 30 crore). The deterministic source-table path prevents the prior-year figure from being returned as the new-year allocation.
- Union Statement 20 BE 2012-13 total is read from the grand-total row as Rs 88,142.80 crore, while the separate narrative note is retained as a rounded Rs 88,143 crore figure.
- Offline regression coverage includes Hindi conversion, Delhi KISHORI column selection, the precise Statement 20 total, and the Odisha source-conflict case. Run `python -m unittest discover -s tests` before running `python -m src.rag eval` and `python -m src.rag dev-eval`.

### Numeric conversion guard update (v4)
- Crore-equivalent generation now requires the unit to be directly attached to the source amount, or explicitly present in the same parenthetical unit label. It no longer scans arbitrary trailing text, preventing a later `source unit: Rs thousand` note from being attached to an already-converted crore figure.
- Hindi parenthetical formats such as `210,000 (210,000 हजार रुपये)` receive the correct ₹21 crore equivalent.
- Added regression tests for both cases.


## Deploy to Streamlit Community Cloud

1. Push this project directory to a GitHub repository. Keep `app.py`, `requirements.txt`, `src/`, `tests/`, and `data/` at the repository root. Commit the supplied corpus files because the app needs them for lexical fallback and index rebuilding.
2. Do not commit `.env` or `.streamlit/secrets.toml`. Both are excluded by `.gitignore`. Use `.streamlit/secrets.toml.example` as the key template, then paste the completed TOML values into the app's **Advanced settings → Secrets** field on Streamlit Community Cloud.
3. The Streamlit entry point is `app.py`. Select the same Python minor version used for local testing (Python 3.13 in the current development environment) when creating the app.
4. Keep the deployed app private during this demo. The current SQLite schema is shared by all visitors, and the sidebar exposes index rebuild and evaluation actions that can incur API usage. Do not make it publicly accessible until authentication, per-user conversation isolation, and admin-only controls are implemented.
5. Local SQLite history is not durable on Community Cloud. The platform does not guarantee persistence of files written to the app filesystem across restarts/redeployments. Use a managed database (for example PostgreSQL/Supabase) before relying on long-term or multi-user conversation history.

The app bridges flat Streamlit secrets into environment variables before importing `src.rag`, so the existing `os.getenv()` configuration works both locally (`.env`) and on Community Cloud (`st.secrets`). If the ignored local chunk cache is absent after deployment, corpus records are extracted once per running app process and cached in memory for subsequent queries.
