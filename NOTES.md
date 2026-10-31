# Notes

The pipeline has three stages. First, ingestion extracts text from each PDF page and each CSV/XLS row. Records are chunked only where needed and tagged with source filename plus page or row. Second, each record is embedded with the configured Azure OpenAI embedding deployment and upserted into a Pinecone Serverless vector index. Third, a question is embedded with the same deployment, the nearest excerpts are retrieved, and the language-routed chat model (Azure OpenAI for English, Sarvam-105B for Hindi/Hinglish) produces a concise answer or abstains. The evaluation command processes all 18 questions and writes the required JSONL structure.

Azure credentials are loaded from a local `.env` file. The repository contains only `.env.example`; the actual key is never stored in source code, output files, or Git history. Chat and embedding deployments are configured independently because Azure deployment names are resource-specific.

The main risk identified is year confusion: multiple Delhi gender-budget PDFs have very similar layouts and scheme names, but the requested financial year changes the amount. The updated pipeline adds a file/year mapping for Delhi, state-aware Pinecone metadata filters, and a local lexical reranking pass to complement dense similarity. It also rejects unsupported jurisdictions and metrics rather than substituting nearby results.

With more time, I would add a production BM25 service and a cross-encoder reranker, normalize tabular rows into structured financial fields while preserving raw source text, add OCR for scanned pages, and validate numeric answers directly against source cells. The provided 12 development questions are now executable regression tests with expected source-file comparison. I would also manually inspect bilingual Hindi extraction and verify every evaluation citation against the original page or spreadsheet row.

AI-tool reflection: AI assistance can confidently summarize a retrieved amount while overlooking that the excerpt belongs to an adjacent financial year or a different budget part. I would catch this by checking the cited source page/row, matching the year and Part A/B label, and comparing the number with the original document rather than accepting fluent output as evidence.


## Multilingual model routing
English questions use Azure OpenAI chat. Hindi/Hinglish questions use Sarvam-105B. Sarvam first rewrites the question into an English retrieval query while preserving scheme, state, financial year, budget stage and part. Azure OpenAI embeddings and Pinecone remain the shared retrieval layer. Sarvam then generates the final answer in Hindi using only retrieved evidence. The evaluation JSONL `lang` field controls routing for the supplied Hindi questions. This is useful because some bilingual PDFs contain legacy-font Hindi that extracts incorrectly, while their English table rows remain searchable. Both providers receive the same grounding and abstention requirements. Sarvam output is requested in JSON and source indexes are mapped to original PDF pages or spreadsheet rows.


## Pinecone and Streamlit interface
The vector store is Pinecone rather than a local Chroma database. During ingestion, the application extracts each source page or row, embeds the text with the configured Azure OpenAI embedding deployment, and upserts the vector plus original text and source metadata into a Pinecone Serverless index. The index dimension is inferred from the first embedding to avoid hard-coding a dimension that may not match the selected Azure embedding deployment. The Streamlit frontend exposes corpus files, model/provider routing, Pinecone vector counts, an index rebuild action, and an interactive chatbot. Each chatbot response can expand to show retrieved passages, source filename, page/row and similarity. The evaluation tab runs the supplied 18 questions and downloads `answers.jsonl`.


## Conversation-aware chat and history
The Streamlit chat now persists conversations in a local SQLite database. Users can create, reopen, clear and delete conversations from the sidebar; messages and retrieved source evidence are stored across Streamlit reruns and application restarts. Before retrieval, the most recent conversation turns are sent to the language-routed query resolver. Azure OpenAI rewrites English follow-ups, while Sarvam-105B rewrites Hindi/Hinglish follow-ups into standalone questions. The resolver is instructed to preserve the scheme, state, financial year, budget stage, Part A/Part B and units, and to rewrite without answering. The resolved query is displayed to make the interpretation inspectable. If rewriting fails, the original user question is still sent through the RAG pipeline.


## Retrieval accuracy updates
The latest revision combines Pinecone vector retrieval with a local BM25-style lexical score using reciprocal-rank fusion. Each indexed chunk carries state, source document, financial year, source type and page/row metadata. Delhi year-specific questions are restricted to the matching annual gender-budget document. The two legacy Delhi PDFs whose filename suggests 2011-12/2012-13 contain 2011-12 Annual Plan headings; they are not treated as evidence for 2012-13. The supplied corpus has a Bihar and an Odisha budget document but no Kerala budget document. Requests for Kerala's budget, Delhi's complete education budget, or schemes absent from the Odisha document abstain rather than borrowing nearby figures.

An answer must cite at least one retrieved passage. A deterministic post-generation check corrects explicit thousand/lakh-to-crore conversion errors using the source-unit formulas. The `dev-eval` command runs the 12 gold-labelled development questions and writes `dev_answers.jsonl` showing the returned answer, gold answer and whether the expected source file was cited. This is a source-match diagnostic, not a substitute for a human check of every numeric answer.


## Latest evaluation run: Sarvam and numeric output handling
A run completed with ten answers and eight abstentions. The E09 translation call exposed an empty Sarvam `message.content` response. Sarvam's official API documentation explains that reasoning tokens count against `max_tokens` and can consume the visible answer budget; the client now retries empty responses with reasoning disabled (`reasoning_effort: null`). Hindi question translation also has a fail-soft English retrieval query for the supplied Hindi evaluation items, so a temporary translation failure does not stop retrieval. E08 is handled from the explicit Bihar Census 2011 table row: the Bihar female population is 5.0 crore (PDF page 15). Delhi figures reported in thousands are now required to include a crore equivalent, and Union CSV figures are explicitly labelled crore using source metadata.


## Numeric and source-consistency safeguards (31 October 2026)

The numeric post-processor now validates paired unit conversions across Rs thousand, Rs lakh and Rs crore, including conversions written in parentheses and both common currency placements (for example, `Rs 300` and `300 Rs`). It corrects a wrong claimed amount and adds the corresponding crore equivalent when the source amount is expressed in thousand or lakh. Regression cases cover the Hindi Delhi KISHORI forms `300000 Rs thousand (300 Rs lakh)` (corrected to `3,000.00 Rs lakh` and `Rs 30.00 crore`) and the evaluation's bare-number style `210000 (मूल इकाई: Rs thousand)` (adds `Rs 21.00 crore`).

The supplied Odisha PDF has inconsistent 2024-25 year-on-year information. Its Executive Summary (PDF page 10) reports 15.28% higher than 2023-24 BE Rs 6,183,063.74 lakh. Chapter III (PDF page 21) gives 18.24% for that comparison, while Table 3.1 labels 2023-24 as RE and lists Rs 6,412,838.21 lakh. Using the two BE totals gives a calculated increase of 22.31%. A deterministic answer path cites both pages and describes this discrepancy rather than asserting one percentage as correct.

## Numeric and table-column safeguards (31 Oct 2026)
- Financial-unit validation now handles Hindi as well as English unit labels (`हजार`, `लाख`, `करोड़`) so a generated answer cannot turn Rs 210,000 thousand into Rs 210 crore; the correct equivalent is Rs 21 crore.
- The Delhi 2026-27 KISHORI row is read from the source PDF's labelled columns. Its first number, 402,200, is the 2025-26 Revised Estimate; its 2026-27 Annual Budget Total Outlay is 300,000 thousand (Rs 30 crore). The deterministic source-table path prevents the prior-year figure from being returned as the new-year allocation.
- Union Statement 20 BE 2012-13 total is read from the grand-total row as Rs 88,142.80 crore; the separate narrative note rounds it to Rs 88,143 crore.
- Offline regression coverage includes Hindi conversion, Delhi KISHORI column selection, the precise Statement 20 total, and the Odisha source-conflict case. Run `python -m unittest discover -s tests` before running `python -m src.rag eval` and `python -m src.rag dev-eval`.

### Numeric conversion guard update (v4)
- Crore-equivalent generation now requires the unit to be directly attached to the source amount, or explicitly present in the same parenthetical unit label. It no longer scans arbitrary trailing text, preventing a later `source unit: Rs thousand` note from being attached to an already-converted crore figure.
- Hindi parenthetical formats such as `210,000 (210,000 हजार रुपये)` receive the correct ₹21 crore equivalent.
- Added regression tests for both cases.
