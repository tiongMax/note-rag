# Gemini PDF RAG baseline

A deliberately naive Phase 1 RAG pipeline that takes one local PDF, stores its
chunks in PostgreSQL with pgvector, and answers questions with Google Gemini.
The application uses LangChain Expression Language (LCEL), not legacy chain
classes.

## Pipeline

1. `PyPDFLoader` extracts the PDF one page at a time.
2. `RecursiveCharacterTextSplitter` creates 700-character chunks with 100
   characters of overlap.
3. `GoogleGenerativeAIEmbeddings` embeds chunks with `gemini-embedding-2`.
4. `langchain_postgres.PGVector` upserts embeddings and metadata into Postgres.
5. The retriever returns the five most similar chunks and prints them.
6. `ChatGoogleGenerativeAI` uses `gemini-3.5-flash` to answer the question.

Each vector includes the source filename, module name, and chunk index. This
baseline intentionally has no re-ranking, hybrid search, citation grounding,
or refusal logic.

## Setup

Create the environment and install the pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start the included local pgvector database:

```powershell
docker compose up -d
```

Create `.env` in the project root:

```dotenv
GOOGLE_API_KEY=your-google-api-key
POSTGRES_CONNECTION_STRING=postgresql+psycopg://langchain:langchain@localhost:6024/langchain
PGVECTOR_COLLECTION=phase1_pdf_rag
EMBEDDING_MODEL=gemini-embedding-2
LLM_MODEL=gemini-3.5-flash
```

Configuration is read directly from `.env` with `python-dotenv`.

## Run

Pass the local PDF as the positional argument:

```powershell
python main.py Shreyas_Puttaraju_Resume.pdf
```

The PDF is loaded, chunked, embedded, and upserted before the interactive
question loop starts.

To skip ingestion and query documents already stored in the configured
PGVector collection, run without a PDF:

```powershell
python main.py
```

Type `exit` or `quit` to stop.

## Limitations

- Re-ingestion overwrites stable chunk positions but does not delete stale
  chunks if a revised PDF becomes shorter.
- Scanned PDFs need OCR before `PyPDFLoader` can extract useful text.
- Retrieval is similarity-only and answers are not citation-grounded.
