# Scope

This implementation includes all supplied source files with supported formats: PDF, CSV, and XLS. PDFs are extracted one page at a time, while CSV and XLS files are represented as row-level records so that scheme names, years, headings, and amounts remain close together. Each record retains its original filename and page or row location. The pipeline indexes extracted text using Azure OpenAI embeddings and stores vectors and metadata in a Pinecone Serverless index.

The implementation does not attempt to normalize every budget table into a common financial schema. This is intentional for the two-hour scope: the corpus contains different state formats, multiple financial years, varying units (rupees, lakh, crore, and thousands), and bilingual documents. Keeping source rows/pages intact reduces the risk of applying an incorrect conversion or merging similar schemes across years.

No OCR is included. Pages with no machine-readable text are skipped. Legacy Hindi PDFs may contain font-encoded text that extracts incorrectly; these are not translated or reconstructed automatically. The XLS file is parsed row-wise using `xlrd`; malformed or unsupported sheets are reported and skipped. CSV rows retain their source columns in a single text record.

The system uses Pinecone dense retrieval together with a local BM25-style lexical ranking pass and reciprocal-rank fusion. Metadata includes source file, state, financial year, page/row and document type. Delhi questions with a specified year are filtered to the matching annual PDF. Bihar and Odisha requests are filtered to the respective state document, preserving multi-year tables within those reports. A full BM25 service and cross-encoder reranker are not included.


Hindi/Hinglish questions are routed to Sarvam-105B for language understanding and final Hindi generation. Sarvam converts the question into an English retrieval query because some bilingual PDFs contain legacy-font Hindi that extracts incorrectly; English table rows can still be retrieved. Azure OpenAI embeddings and Pinecone remain shared across languages, while English questions continue to use Azure OpenAI chat. Evaluation language labels control deterministic routing.


Pinecone Serverless is used for persistent vector storage. The Streamlit UI exposes ingestion, index status, a multilingual chatbot, retrieved evidence and evaluation execution. This makes the retrieval and answer flow inspectable instead of hiding it behind a command-line interface.


Conversation-aware follow-ups are supported using a short recent-turn window. Query rewriting is performed by the relevant language model before Pinecone retrieval. Conversation messages and source evidence are persisted locally in SQLite so chats can be resumed after the Streamlit process restarts. This is a single-application local history store, not multi-user authentication or cloud-synchronized history.


The supplied corpus does not contain a Kerala budget document or Delhi's full department-level education budget. The document named `gender_budget_2012-13.pdf` exposes 2011-12 Annual Plan columns, not a verified 2012-13 allocation. These unsupported requests abstain. The Odisha PDF does not contain a Mahila Samriddhi Yojana entry, so the system does not infer its allocation. Explicit thousand/lakh-to-crore conversions are checked after generation. All cited answers still require visual verification against the source PDF page or spreadsheet row.


Sarvam short-form JSON calls disable reasoning by default and retry an empty visible response with reasoning disabled, because reasoning tokens count against the completion token budget. Hindi evaluation questions also have query-normalization fallbacks so translation API interruptions do not prevent retrieval. Bihar's Census 2011 female-population table is extracted from its cited PDF page. Source-unit metadata is carried into Pinecone and the answer prompt so thousand/lakh/crore amounts are clearly labelled.
