"""Command-line entry point for the FastAPI application."""

import uvicorn


def main() -> int:
    uvicorn.run(
        "note_rag.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
