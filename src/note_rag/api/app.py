"""FastAPI application factory."""

import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from starlette.concurrency import run_in_threadpool

from note_rag.api.models import (
    ChunkTextRequest,
    ChunkTextResponse,
    DocumentResponse,
    IngestionJobResponse,
    IngestionResponse,
    StoredChunkResponse,
)
from note_rag.api.settings import ApiSettings, api_settings
from note_rag.chunking import RegexTokenCounter, TokenChunker
from note_rag.ingest import IngestionPipeline, LocalFileStorage, ParserRegistry
from note_rag.ingest.errors import UnsupportedDocumentTypeError
from note_rag.persistence import (
    ChunkRepository,
    Database,
    DocumentRepository,
    IngestionJobRepository,
)


def create_app(
    app_settings: ApiSettings = api_settings,
    *,
    database: Database | None = None,
    storage: LocalFileStorage | None = None,
) -> FastAPI:
    """Build an application without starting network services."""

    resolved_database = database or Database()
    resolved_storage = storage or LocalFileStorage(app_settings.storage_path)
    parser_registry = ParserRegistry()
    pipeline = IngestionPipeline(
        resolved_database,
        resolved_storage,
        parser_registry=parser_registry,
        chunker=TokenChunker(
            chunk_size=app_settings.chunk_size,
            chunk_overlap=app_settings.chunk_overlap,
        ),
    )
    app = FastAPI(
        title=app_settings.app_name,
        version="0.1.0",
        description="Phase 3: persistent document ingestion and chunking.",
    )
    app.state.database = resolved_database
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

    @app.post(
        "/api/v1/documents",
        response_model=IngestionResponse,
        status_code=201,
        tags=["documents"],
    )
    async def upload_document(
        response: Response,
        file: UploadFile = File(...),
    ) -> IngestionResponse:
        filename = Path(file.filename or "").name
        if not filename:
            raise HTTPException(status_code=422, detail="filename is required")
        try:
            parser_registry.get(filename)
        except UnsupportedDocumentTypeError as error:
            raise HTTPException(status_code=415, detail=str(error)) from error

        content = await file.read(app_settings.max_upload_bytes + 1)
        await file.close()
        if not content:
            raise HTTPException(status_code=422, detail="uploaded file is empty")
        if len(content) > app_settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"file exceeds the {app_settings.max_upload_bytes}-byte limit"
                ),
            )

        result = await run_in_threadpool(
            pipeline.ingest,
            filename=filename,
            media_type=file.content_type or "application/octet-stream",
            content=content,
        )
        if result.duplicate:
            response.status_code = 200
        elif result.error_message is not None:
            response.status_code = 422
        return IngestionResponse(
            document_id=result.document_id,
            job_id=result.job_id,
            status=result.status,
            duplicate=result.duplicate,
            chunk_count=result.chunk_count,
            token_count=result.token_count,
            error_message=result.error_message,
        )

    @app.get(
        "/api/v1/documents",
        response_model=list[DocumentResponse],
        tags=["documents"],
    )
    def list_documents() -> list[DocumentResponse]:
        with resolved_database.session() as session:
            documents = DocumentRepository(session).list()
            return [DocumentResponse.model_validate(item) for item in documents]

    @app.get(
        "/api/v1/documents/{document_id}",
        response_model=DocumentResponse,
        tags=["documents"],
    )
    def get_document(document_id: uuid.UUID) -> DocumentResponse:
        with resolved_database.session() as session:
            document = DocumentRepository(session).get(document_id)
            if document is None:
                raise HTTPException(status_code=404, detail="document not found")
            return DocumentResponse.model_validate(document)

    @app.get(
        "/api/v1/documents/{document_id}/chunks",
        response_model=list[StoredChunkResponse],
        tags=["documents"],
    )
    def list_document_chunks(
        document_id: uuid.UUID,
    ) -> list[StoredChunkResponse]:
        with resolved_database.session() as session:
            if DocumentRepository(session).get(document_id) is None:
                raise HTTPException(status_code=404, detail="document not found")
            chunks = ChunkRepository(session).list_for_document(document_id)
            return [StoredChunkResponse.model_validate(item) for item in chunks]

    @app.get(
        "/api/v1/ingestion-jobs/{job_id}",
        response_model=IngestionJobResponse,
        tags=["ingestion"],
    )
    def get_ingestion_job(job_id: uuid.UUID) -> IngestionJobResponse:
        with resolved_database.session() as session:
            job = IngestionJobRepository(session).get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="ingestion job not found")
            return IngestionJobResponse.model_validate(job)

    return app


app = create_app()
