"""FastAPI application factory."""

import json
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Response, UploadFile, status
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse
from starlette.staticfiles import StaticFiles

from note_rag.api.models import (
    ChatMessageResponse,
    ChatRequest,
    ChatResponse,
    ChunkTextRequest,
    ChunkTextResponse,
    ContextRequest,
    ContextResponse,
    ConversationDetailResponse,
    ConversationResponse,
    DocumentResponse,
    IndexingResponse,
    IngestionJobResponse,
    IngestionResponse,
    SearchRequest,
    SearchResponse,
    StoredChunkResponse,
)
from note_rag.api.settings import ApiSettings, api_settings
from note_rag.chat import (
    ChatOptions,
    ChatProvider,
    ChatService,
    GeminiChatProvider,
)
from note_rag.chunking import RegexTokenCounter, TokenChunker
from note_rag.context import ContextBuilder, LexicalReranker, Reranker
from note_rag.embeddings import (
    GeminiEmbeddingProvider,
    IndexingService,
    QueryEmbeddingProvider,
)
from note_rag.ingest import (
    IngestionPipeline,
    IngestionWorker,
    LocalFileStorage,
    ParserRegistry,
)
from note_rag.ingest.errors import UnsupportedDocumentTypeError
from note_rag.persistence import (
    ChatMessageRepository,
    ChunkRepository,
    ConversationRepository,
    Database,
    DocumentRepository,
    IngestionJobRepository,
)
from note_rag.retrieval import RetrievalService, SearchFilters


def create_app(
    app_settings: ApiSettings = api_settings,
    *,
    database: Database | None = None,
    storage: LocalFileStorage | None = None,
    embedding_provider: QueryEmbeddingProvider | None = None,
    reranker: Reranker | None = None,
    chat_provider: ChatProvider | None = None,
) -> FastAPI:
    """Build an application without starting network services."""

    resolved_database = database or Database()
    resolved_storage = storage or LocalFileStorage(app_settings.storage_path)
    parser_registry = ParserRegistry()
    token_counter = RegexTokenCounter()
    resolved_embedding_provider = (
        embedding_provider
        or GeminiEmbeddingProvider(
            app_settings.embedding_model,
            api_key=app_settings.gemini_api_key,
            expected_dimension=app_settings.embedding_dimension,
        )
    )
    indexing_service = IndexingService(
        resolved_database,
        resolved_embedding_provider,
        batch_size=app_settings.embedding_batch_size,
    )
    retrieval_service = RetrievalService(
        resolved_database,
        resolved_embedding_provider,
        candidate_multiplier=app_settings.retrieval_candidate_multiplier,
        rrf_k=app_settings.retrieval_rrf_k,
    )
    context_builder = ContextBuilder(
        retrieval_service,
        reranker or LexicalReranker(token_counter),
        token_counter=token_counter,
    )
    resolved_chat_provider = chat_provider or GeminiChatProvider(
        app_settings.chat_model,
        api_key=app_settings.gemini_api_key,
        temperature=app_settings.chat_temperature,
        max_output_tokens=app_settings.chat_max_output_tokens,
    )
    chat_service = ChatService(
        resolved_database,
        context_builder,
        resolved_chat_provider,
        token_counter=token_counter,
        history_max_messages=app_settings.chat_history_max_messages,
        history_max_tokens=app_settings.chat_history_max_tokens,
    )
    pipeline = IngestionPipeline(
        resolved_database,
        resolved_storage,
        parser_registry=parser_registry,
        chunker=TokenChunker(
            chunk_size=app_settings.chunk_size,
            chunk_overlap=app_settings.chunk_overlap,
        ),
    )
    ingestion_worker = IngestionWorker(
        resolved_database,
        pipeline,
        indexing_service,
        max_attempts=app_settings.worker_max_attempts,
        retry_backoff_seconds=app_settings.worker_retry_backoff_seconds,
        poll_interval_seconds=app_settings.worker_poll_interval_seconds,
        lease_timeout_seconds=app_settings.worker_lease_timeout_seconds,
    )
    stop_event = threading.Event()
    worker_thread: threading.Thread | None = None

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        nonlocal worker_thread
        if app_settings.background_worker_enabled:
            worker_thread = threading.Thread(
                target=ingestion_worker.run_forever,
                args=(stop_event,),
                name="note-rag-ingestion-worker",
                daemon=True,
            )
            worker_thread.start()
            application.state.worker_thread = worker_thread
        try:
            yield
        finally:
            stop_event.set()
            if worker_thread is not None:
                worker_thread.join(timeout=5)

    app = FastAPI(
        title=app_settings.app_name,
        version="0.1.0",
        description="A compact ingestion, retrieval, and grounded chat service.",
        lifespan=lifespan,
    )
    app.state.database = resolved_database
    app.state.ingestion_worker = ingestion_worker

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
            pipeline.enqueue,
            filename=filename,
            media_type=file.content_type or "application/octet-stream",
            content=content,
        )
        if result.duplicate:
            response.status_code = 200
        elif app_settings.background_worker_enabled:
            response.status_code = 202
        elif result.job_id is not None:
            await run_in_threadpool(ingestion_worker.run_job, result.job_id)
        with resolved_database.session() as session:
            persisted_document = DocumentRepository(session).get(result.document_id)
            if persisted_document is None:
                raise HTTPException(
                    status_code=500,
                    detail="document was not persisted",
                )
            if (
                not app_settings.background_worker_enabled
                and persisted_document.status.value == "failed"
            ):
                response.status_code = 422
        return IngestionResponse(
            document_id=result.document_id,
            job_id=result.job_id,
            status=persisted_document.status,
            duplicate=result.duplicate,
            chunk_count=persisted_document.chunk_count,
            token_count=persisted_document.token_count,
            error_message=persisted_document.error_message,
            indexing_status=persisted_document.indexing_status,
            indexing_error=persisted_document.indexing_error,
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

    @app.delete(
        "/api/v1/documents/{document_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        tags=["documents"],
    )
    def delete_document(document_id: uuid.UUID) -> Response:
        storage_uri: str | None
        with resolved_database.session() as session:
            repository = DocumentRepository(session)
            document = repository.get(document_id)
            if document is None:
                raise HTTPException(status_code=404, detail="document not found")
            storage_uri = document.storage_uri
            repository.delete(document)
        if storage_uri is not None:
            resolved_storage.delete(storage_uri)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

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

    @app.post(
        "/api/v1/documents/{document_id}/index",
        response_model=IndexingResponse,
        tags=["documents"],
    )
    async def reindex_document(document_id: uuid.UUID) -> IndexingResponse:
        try:
            result = await run_in_threadpool(
                indexing_service.index_document,
                document_id,
                force=True,
            )
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return IndexingResponse(
            document_id=result.document_id,
            status=result.status,
            indexed_chunks=result.indexed_chunks,
            embedding_model=result.embedding_model,
            error_message=result.error_message,
        )

    @app.post(
        "/api/v1/retrieval/search",
        response_model=SearchResponse,
        tags=["retrieval"],
    )
    async def search_chunks(request: SearchRequest) -> SearchResponse:
        filters = SearchFilters(
            document_ids=tuple(request.filters.document_ids),
            filenames=tuple(request.filters.filenames),
            media_types=tuple(request.filters.media_types),
            source_metadata=request.filters.source_metadata,
        )
        try:
            result = await run_in_threadpool(
                retrieval_service.search,
                request.query,
                mode=request.mode,
                top_k=request.top_k,
                vector_weight=request.vector_weight,
                filters=filters,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return SearchResponse.model_validate(result, from_attributes=True)

    @app.post(
        "/api/v1/retrieval/context",
        response_model=ContextResponse,
        tags=["retrieval"],
    )
    async def build_context(request: ContextRequest) -> ContextResponse:
        filters = SearchFilters(
            document_ids=tuple(request.filters.document_ids),
            filenames=tuple(request.filters.filenames),
            media_types=tuple(request.filters.media_types),
            source_metadata=request.filters.source_metadata,
        )
        try:
            result = await run_in_threadpool(
                context_builder.build,
                request.query,
                mode=request.mode,
                candidate_k=(
                    request.candidate_k or app_settings.context_candidate_k
                ),
                max_chunks=(
                    request.max_chunks or app_settings.context_max_chunks
                ),
                max_context_tokens=(
                    request.max_context_tokens
                    or app_settings.context_max_tokens
                ),
                vector_weight=request.vector_weight,
                rerank=request.rerank,
                rerank_weight=(
                    request.rerank_weight
                    if request.rerank_weight is not None
                    else app_settings.rerank_weight
                ),
                filters=filters,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return ContextResponse.model_validate(result, from_attributes=True)

    def chat_options(request: ChatRequest) -> ChatOptions:
        return ChatOptions(
            mode=request.mode,
            candidate_k=request.candidate_k or app_settings.context_candidate_k,
            max_chunks=request.max_chunks or app_settings.context_max_chunks,
            max_context_tokens=(
                request.max_context_tokens or app_settings.context_max_tokens
            ),
            vector_weight=request.vector_weight,
            rerank=request.rerank,
            rerank_weight=(
                request.rerank_weight
                if request.rerank_weight is not None
                else app_settings.rerank_weight
            ),
            filters=SearchFilters(
                document_ids=tuple(request.filters.document_ids),
                filenames=tuple(request.filters.filenames),
                media_types=tuple(request.filters.media_types),
                source_metadata=request.filters.source_metadata,
            ),
        )

    @app.post(
        "/api/v1/chat",
        response_model=ChatResponse,
        tags=["chat"],
    )
    async def chat(request: ChatRequest) -> ChatResponse:
        try:
            result = await run_in_threadpool(
                chat_service.ask,
                request.query,
                conversation_id=request.conversation_id,
                options=chat_options(request),
            )
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return ChatResponse.model_validate(result, from_attributes=True)

    @app.post(
        "/api/v1/chat/stream",
        tags=["chat"],
    )
    async def stream_chat(request: ChatRequest) -> StreamingResponse:
        def stream_events():
            try:
                for event in chat_service.stream(
                    request.query,
                    conversation_id=request.conversation_id,
                    options=chat_options(request),
                ):
                    payload = json.dumps(event.data, ensure_ascii=False)
                    yield f"event: {event.event}\ndata: {payload}\n\n"
            except Exception as error:
                payload = json.dumps(
                    {"detail": str(error) or error.__class__.__name__},
                    ensure_ascii=False,
                )
                yield f"event: error\ndata: {payload}\n\n"

        return StreamingResponse(
            stream_events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get(
        "/api/v1/conversations",
        response_model=list[ConversationResponse],
        tags=["chat"],
    )
    def list_conversations() -> list[ConversationResponse]:
        with resolved_database.session() as session:
            conversations = ConversationRepository(session).list()
            return [
                ConversationResponse(
                    id=conversation.id,
                    title=conversation.title,
                    message_count=len(conversation.messages),
                    created_at=conversation.created_at,
                    updated_at=conversation.updated_at,
                )
                for conversation in conversations
            ]

    @app.get(
        "/api/v1/conversations/{conversation_id}",
        response_model=ConversationDetailResponse,
        tags=["chat"],
    )
    def get_conversation(
        conversation_id: uuid.UUID,
    ) -> ConversationDetailResponse:
        with resolved_database.session() as session:
            conversation = ConversationRepository(session).get(conversation_id)
            if conversation is None:
                raise HTTPException(
                    status_code=404,
                    detail="conversation not found",
                )
            messages = ChatMessageRepository(session).list_for_conversation(
                conversation_id
            )
            return ConversationDetailResponse(
                id=conversation.id,
                title=conversation.title,
                message_count=len(messages),
                created_at=conversation.created_at,
                updated_at=conversation.updated_at,
                messages=[
                    ChatMessageResponse.model_validate(message)
                    for message in messages
                ],
            )

    frontend_dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
    if frontend_dist.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=frontend_dist, html=True),
            name="operator-interface",
        )

    return app


app = create_app()
