"""FastAPI application factory."""

from fastapi import FastAPI, HTTPException

from note_rag.api.models import ChunkTextRequest, ChunkTextResponse
from note_rag.api.settings import ApiSettings, api_settings
from note_rag.chunking import RegexTokenCounter, TokenChunker


def create_app(app_settings: ApiSettings = api_settings) -> FastAPI:
    """Build an application without starting network services."""

    app = FastAPI(
        title=app_settings.app_name,
        version="0.1.0",
        description="Phase 1: deterministic token-aware chunking.",
    )
    token_counter = RegexTokenCounter()

    @app.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": app_settings.app_name,
            "environment": app_settings.app_environment,
        }

    @app.post("/api/v1/chunks", response_model=ChunkTextResponse, tags=["chunking"])
    async def chunk_text(request: ChunkTextRequest) -> ChunkTextResponse:
        chunk_size = request.chunk_size or app_settings.chunk_size
        chunk_overlap = (
            request.chunk_overlap
            if request.chunk_overlap is not None
            else app_settings.chunk_overlap
        )
        try:
            chunker = TokenChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                token_counter=token_counter,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        return ChunkTextResponse(
            token_count=token_counter.count(request.text),
            chunks=chunker.chunk(request.text, source_id=request.source_id),
        )

    return app


app = create_app()
