# Gemini PDF RAG: Phase 2

This project extends the Phase 1 PDF RAG pipeline with configurable dense,
hybrid, and cross-encoder-reranked retrieval. The Phase 1 PDF loading,
700-character chunking, Gemini embedding, stable-ID generation, and PGVector
ingestion logic is unchanged.

## Pipeline

1. `PyPDFLoader` extracts the PDF one page at a time.
2. `RecursiveCharacterTextSplitter` creates 700-character chunks with 100
   characters of overlap.
3. `GoogleGenerativeAIEmbeddings` embeds chunks with `gemini-embedding-2`.
4. `langchain_postgres.PGVector` upserts embeddings and metadata into Postgres.
5. Retrieval runs in one of three configurations:
   - `dense-only`: five PGVector similarity results.
   - `hybrid`: weighted reciprocal-rank fusion of 20 dense and 20 BM25 results,
     reduced to five.
   - `hybrid+rerank`: the top 20 hybrid results are scored by
     `cross-encoder/ms-marco-MiniLM-L6-v2` and reduced to five.
6. `ChatGoogleGenerativeAI` uses `gemini-3.5-flash` to answer from those chunks.

LangChain's `BM25Retriever` is built at startup from the same stored PGVector
rows; it does not re-load or re-chunk the original PDFs. The first reranked query
downloads and caches the cross-encoder model if it is not already available
locally.

## Setup

Create the environment and install the application with its development tools:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Start the included local pgvector database:

```powershell
docker compose up -d
```

Copy `.env.example` to `.env`, then add your API key:

```powershell
Copy-Item .env.example .env
```

## Run

Use the installed command in the examples below. `python main.py` remains
available as a compatibility launcher.

Optionally pass one PDF to run the ingestion path before chat:

```powershell
note-rag documents/Shreyas_Puttaraju_Resume.pdf --strategy hybrid+rerank
```

Pass multiple PDFs to batch ingest them sequentially before chat:

```powershell
note-rag notes.pdf handbook.pdf report.pdf --strategy hybrid+rerank
```

Query an existing collection with any retrieval strategy:

```powershell
note-rag --strategy dense-only
note-rag --strategy hybrid
note-rag --strategy hybrid+rerank
```

Type `exit` or `quit` to stop.

## Evaluation

See [EVALUATION.md](EVALUATION.md) for dataset format, metrics, commands, and
generated reports.

## Project structure

```text
note-rag/
├── src/note_rag/         # Installable application package
│   ├── app/              # CLI and evaluation workflows
│   ├── evaluation/       # Metrics, evaluators, models, and reports
│   ├── rag/              # Answer generation and LLM judging
│   ├── retrieval/        # Strategies, pipelines, and factories
│   ├── config.py         # Environment-backed settings
│   ├── ingestion.py      # PDF loading, chunking, and upserts
│   └── __main__.py       # `python -m note_rag` entry point
├── tests/                # Unit tests
├── evaluations/
│   ├── datasets/         # Versioned evaluation inputs
│   └── results/          # Generated benchmark reports
├── documents/            # Local source PDFs (git-ignored)
├── .env.example          # Safe configuration template
├── docker-compose.yml    # Local pgvector service
└── pyproject.toml        # Package, dependencies, and tool configuration
```

Run the local quality checks with:

```powershell
ruff check .
pyright
pytest
```

## Limitations

- Re-ingestion overwrites stable chunk positions but does not delete stale
  chunks if a revised PDF becomes shorter.
- Scanned PDFs need OCR before `PyPDFLoader` can extract useful text.
- The in-memory LangChain BM25 retriever is rebuilt from the stored collection
  on each process start.
- The English MS MARCO reranker may be a poor fit for non-English corpora.
- Answers are not citation-grounded and out-of-corpus questions do not yet have
  explicit refusal logic; the end-to-end evaluation measures this gap.
- LLM-as-judge scores are model-dependent and complement, rather than replace,
  human review.
