"""FastAPI application factory."""

import hashlib
import json
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Response, UploadFile, status
from sqlalchemy import text
from starlette.concurrency import run_in_threadpool
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import PlainTextResponse, StreamingResponse
from starlette.staticfiles import StaticFiles

from note_rag import __version__
from note_rag.api.errors import install_error_handlers
from note_rag.api.middleware import install_http_middleware
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
    CourseCreateRequest,
    CourseDocumentsRequest,
    CourseResponse,
    CourseUpdateRequest,
    DocumentResponse,
    GenerationJobResponse,
    GenerationRequest,
    IndexingResponse,
    IngestionJobResponse,
    IngestionResponse,
    SearchRequest,
    SearchResponse,
    SourcePassageResponse,
    StoredChunkResponse,
    StudyItemResponse,
    StudyItemUpdateRequest,
    TopicCreateRequest,
    TopicMergeRequest,
    TopicResponse,
    TopicUpdateRequest,
)
from note_rag.api.observability import MetricsRegistry, configure_logging
from note_rag.api.settings import ApiSettings, api_settings
from note_rag.chat import (
    ChatOptions,
    ChatProvider,
    ChatService,
    GeminiChatProvider,
)
from note_rag.chunking import (
    Chunker,
    RecursiveChunker,
    RegexTokenCounter,
    TokenChunker,
)
from note_rag.context import (
    ContextBuilder,
    CrossEncoderReranker,
    LexicalReranker,
    Reranker,
)
from note_rag.embeddings import (
    DeterministicEmbeddingProvider,
    GeminiEmbeddingProvider,
    IndexingService,
    QueryEmbeddingProvider,
)
from note_rag.generation import (
    PROMPT_VERSION,
    GenerationWorker,
    LearningMaterialService,
)
from note_rag.ingest import (
    IngestionPipeline,
    IngestionWorker,
    LocalFileStorage,
    ParserRegistry,
)
from note_rag.ingest.errors import UnsupportedDocumentTypeError
from note_rag.persistence import (
    ApprovalStatus,
    ChatMessageRepository,
    ChunkRepository,
    ConversationRepository,
    Course,
    CourseRepository,
    Database,
    DocumentRepository,
    GenerationJob,
    GenerationJobRepository,
    GenerationJobStatus,
    GenerationKind,
    IngestionJobRepository,
    StudyItem,
    StudyItemRepository,
    Topic,
    TopicRepository,
    TopicState,
)
from note_rag.retrieval import (
    PersistentRetrievalCache,
    RetrievalService,
    SearchFilters,
)


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

    configure_logging(
        app_settings.log_level,
        json_logs=app_settings.json_logs,
    )
    metrics = MetricsRegistry()
    owns_database = database is None
    resolved_database = database or Database()
    resolved_storage = storage or LocalFileStorage(app_settings.storage_path)
    parser_registry = ParserRegistry()
    token_counter = RegexTokenCounter()
    resolved_embedding_provider = embedding_provider or (
        DeterministicEmbeddingProvider(
            dimension=app_settings.embedding_dimension,
            delay_ms=app_settings.benchmark_embedding_delay_ms,
        )
        if app_settings.embedding_backend == "deterministic"
        else GeminiEmbeddingProvider(
            app_settings.embedding_model,
            api_key=app_settings.gemini_api_key,
            expected_dimension=app_settings.embedding_dimension,
        )
    )
    retrieval_cache = PersistentRetrievalCache(
        resolved_database,
        enabled=app_settings.cache_enabled,
        embedding_enabled=app_settings.embedding_cache_enabled,
        retrieval_enabled=app_settings.retrieval_cache_enabled,
        embedding_ttl_seconds=app_settings.embedding_cache_ttl_seconds,
        retrieval_ttl_seconds=app_settings.retrieval_cache_ttl_seconds,
        metrics=metrics,
    )
    indexing_service = IndexingService(
        resolved_database,
        resolved_embedding_provider,
        batch_size=app_settings.embedding_batch_size,
        cache=retrieval_cache,
    )
    retrieval_service = RetrievalService(
        resolved_database,
        resolved_embedding_provider,
        candidate_multiplier=app_settings.retrieval_candidate_multiplier,
        rrf_k=app_settings.retrieval_rrf_k,
        cache=retrieval_cache,
    )
    resolved_reranker = reranker
    if resolved_reranker is None:
        resolved_reranker = (
            CrossEncoderReranker(
                app_settings.cross_encoder_model,
                device=app_settings.cross_encoder_device or None,
                batch_size=app_settings.cross_encoder_batch_size,
            )
            if app_settings.reranker_backend == "cross_encoder"
            else LexicalReranker(token_counter)
        )
    context_builder = ContextBuilder(
        retrieval_service,
        resolved_reranker,
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
    learning_material_service = LearningMaterialService(
        resolved_database, resolved_chat_provider
    )
    generation_worker = GenerationWorker(
        resolved_database,
        learning_material_service,
        max_attempts=app_settings.worker_max_attempts,
        poll_interval=app_settings.worker_poll_interval_seconds,
    )
    chunker_class: type[TokenChunker] | type[RecursiveChunker] = (
        RecursiveChunker
        if app_settings.chunking_strategy == "recursive"
        else TokenChunker
    )
    pipeline = IngestionPipeline(
        resolved_database,
        resolved_storage,
        parser_registry=parser_registry,
        chunker=chunker_class(
            chunk_size=app_settings.chunk_size,
            chunk_overlap=app_settings.chunk_overlap,
            token_counter=token_counter,
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
    worker_threads: list[threading.Thread] = []

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if app_settings.background_worker_enabled:
            worker_threads.extend(
                [
                    threading.Thread(
                        target=ingestion_worker.run_forever,
                        args=(stop_event,),
                        name="note-rag-ingestion-worker",
                        daemon=True,
                    ),
                    threading.Thread(
                        target=generation_worker.run_forever,
                        args=(stop_event,),
                        name="note-rag-generation-worker",
                        daemon=True,
                    ),
                ]
            )
            for worker_thread in worker_threads:
                worker_thread.start()
            application.state.worker_threads = worker_threads
        try:
            yield
        finally:
            stop_event.set()
            for worker_thread in worker_threads:
                worker_thread.join(timeout=5)
            if owns_database:
                resolved_database.dispose()

    production = app_settings.app_environment.lower() in {"production", "prod"}
    app = FastAPI(
        title=app_settings.app_name,
        version=__version__,
        description="A compact ingestion, retrieval, and grounded chat service.",
        lifespan=lifespan,
        docs_url=None if production else "/docs",
        redoc_url=None if production else "/redoc",
        openapi_url=None if production else "/openapi.json",
    )
    app.state.database = resolved_database
    app.state.ingestion_worker = ingestion_worker
    app.state.generation_worker = generation_worker
    app.state.metrics = metrics
    app.state.retrieval_cache = retrieval_cache
    install_error_handlers(app)
    install_http_middleware(app, app_settings, metrics)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=list(app_settings.allowed_hosts),
    )
    if app_settings.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(app_settings.allowed_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=[
                "Authorization",
                "Content-Type",
                "X-API-Key",
                "X-Request-ID",
            ],
            expose_headers=[
                "X-Request-ID",
                "Retry-After",
                "X-Embedding-Cache",
                "X-Retrieval-Cache",
                "X-Corpus-Version",
            ],
        )

    @app.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": app_settings.app_name,
            "environment": app_settings.app_environment,
        }

    @app.get("/health/ready", tags=["system"])
    async def readiness() -> dict[str, str]:
        try:
            await run_in_threadpool(_check_database, resolved_database)
        except Exception as error:
            raise HTTPException(
                status_code=503,
                detail="database is not ready",
            ) from error
        return {"status": "ready", "database": "ok"}

    if app_settings.metrics_enabled:

        @app.get(
            "/metrics",
            response_class=PlainTextResponse,
            include_in_schema=False,
        )
        async def prometheus_metrics() -> str:
            return metrics.render()

    @app.post("/api/v1/chunks", response_model=ChunkTextResponse, tags=["chunking"])
    async def chunk_text(request: ChunkTextRequest) -> ChunkTextResponse:
        chunk_size = request.chunk_size or app_settings.chunk_size
        chunk_overlap = (
            request.chunk_overlap
            if request.chunk_overlap is not None
            else app_settings.chunk_overlap
        )
        try:
            chunker: Chunker = chunker_class(
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

    def course_response(course: Course) -> CourseResponse:
        return CourseResponse(
            id=course.id,
            title=course.title,
            description=course.description,
            document_ids=[link.document_id for link in course.document_links],
            topic_count=len(course.topics),
            created_at=course.created_at,
            updated_at=course.updated_at,
        )

    def topic_response(topic: Topic) -> TopicResponse:
        return TopicResponse(
            id=topic.id,
            course_id=topic.course_id,
            parent_id=topic.parent_id,
            title=topic.title,
            description=topic.description,
            position=topic.position,
            state=topic.state,
            source_chunk_ids=[source.chunk_id for source in topic.sources],
            created_at=topic.created_at,
            updated_at=topic.updated_at,
        )

    @app.post(
        "/api/v1/courses",
        response_model=CourseResponse,
        status_code=201,
        tags=["courses"],
    )
    def create_course(request: CourseCreateRequest) -> CourseResponse:
        title = request.title.strip()
        if not title:
            raise HTTPException(status_code=422, detail="course title cannot be blank")
        with resolved_database.session() as session:
            course = CourseRepository(session).add(
                Course(title=title, description=request.description.strip())
            )
            return course_response(course)

    @app.get("/api/v1/courses", response_model=list[CourseResponse], tags=["courses"])
    def list_courses() -> list[CourseResponse]:
        with resolved_database.session() as session:
            return [
                course_response(course) for course in CourseRepository(session).list()
            ]

    @app.get(
        "/api/v1/courses/{course_id}", response_model=CourseResponse, tags=["courses"]
    )
    def get_course(course_id: uuid.UUID) -> CourseResponse:
        with resolved_database.session() as session:
            course = CourseRepository(session).get(course_id)
            if course is None:
                raise HTTPException(status_code=404, detail="course not found")
            return course_response(course)

    @app.patch(
        "/api/v1/courses/{course_id}", response_model=CourseResponse, tags=["courses"]
    )
    def update_course(
        course_id: uuid.UUID, request: CourseUpdateRequest
    ) -> CourseResponse:
        with resolved_database.session() as session:
            course = CourseRepository(session).get(course_id)
            if course is None:
                raise HTTPException(status_code=404, detail="course not found")
            if request.title is not None:
                title = request.title.strip()
                if not title:
                    raise HTTPException(
                        status_code=422, detail="course title cannot be blank"
                    )
                course.title = title
            if request.description is not None:
                course.description = request.description.strip()
            session.flush()
            return course_response(course)

    @app.delete("/api/v1/courses/{course_id}", status_code=204, tags=["courses"])
    def delete_course(course_id: uuid.UUID) -> Response:
        with resolved_database.session() as session:
            repository = CourseRepository(session)
            course = repository.get(course_id)
            if course is None:
                raise HTTPException(status_code=404, detail="course not found")
            repository.delete(course)
        return Response(status_code=204)

    @app.post(
        "/api/v1/courses/{course_id}/documents",
        response_model=CourseResponse,
        tags=["courses"],
    )
    def attach_course_documents(
        course_id: uuid.UUID, request: CourseDocumentsRequest
    ) -> CourseResponse:
        with resolved_database.session() as session:
            repository = CourseRepository(session)
            course = repository.get(course_id)
            if course is None:
                raise HTTPException(status_code=404, detail="course not found")
            documents = []
            missing = []
            document_repository = DocumentRepository(session)
            for document_id in request.document_ids:
                document = document_repository.get(document_id)
                if document is None:
                    missing.append(str(document_id))
                else:
                    documents.append(document)
            if missing:
                raise HTTPException(
                    status_code=404, detail=f"documents not found: {', '.join(missing)}"
                )
            for document in documents:
                repository.attach_document(course, document)
            return course_response(course)

    @app.delete(
        "/api/v1/courses/{course_id}/documents/{document_id}",
        status_code=204,
        tags=["courses"],
    )
    def detach_course_document(
        course_id: uuid.UUID, document_id: uuid.UUID
    ) -> Response:
        with resolved_database.session() as session:
            repository = CourseRepository(session)
            if repository.get(course_id) is None:
                raise HTTPException(status_code=404, detail="course not found")
            if not repository.detach_document(course_id, document_id):
                raise HTTPException(
                    status_code=404, detail="document is not attached to this course"
                )
        return Response(status_code=204)

    @app.get(
        "/api/v1/courses/{course_id}/topics",
        response_model=list[TopicResponse],
        tags=["topics"],
    )
    def list_topics(course_id: uuid.UUID) -> list[TopicResponse]:
        with resolved_database.session() as session:
            if CourseRepository(session).get(course_id) is None:
                raise HTTPException(status_code=404, detail="course not found")
            return [
                topic_response(topic)
                for topic in TopicRepository(session).list_for_course(course_id)
            ]

    @app.get(
        "/api/v1/courses/{course_id}/topics/{topic_id}/sources",
        response_model=list[SourcePassageResponse],
        tags=["topics"],
    )
    def list_topic_sources(
        course_id: uuid.UUID, topic_id: uuid.UUID
    ) -> list[SourcePassageResponse]:
        with resolved_database.session() as session:
            topic = TopicRepository(session).get(topic_id)
            if topic is None or topic.course_id != course_id:
                raise HTTPException(status_code=404, detail="topic not found")
            return [
                SourcePassageResponse(
                    chunk_id=link.chunk.id,
                    document_id=link.chunk.document_id,
                    filename=link.chunk.document.filename,
                    position=link.chunk.position,
                    text=link.chunk.text,
                )
                for link in topic.sources
            ]

    @app.post(
        "/api/v1/courses/{course_id}/topics",
        response_model=TopicResponse,
        status_code=201,
        tags=["topics"],
    )
    def create_topic(
        course_id: uuid.UUID, request: TopicCreateRequest
    ) -> TopicResponse:
        title = request.title.strip()
        if not title:
            raise HTTPException(status_code=422, detail="topic title cannot be blank")
        with resolved_database.session() as session:
            course = CourseRepository(session).get(course_id)
            if course is None:
                raise HTTPException(status_code=404, detail="course not found")
            repository = TopicRepository(session)
            parent = repository.get(request.parent_id) if request.parent_id else None
            if request.parent_id and parent is None:
                raise HTTPException(status_code=404, detail="parent topic not found")
            try:
                topic = repository.add(
                    course,
                    title=title,
                    description=request.description.strip(),
                    parent=parent,
                    state=request.state,
                    position=request.position,
                )
                for chunk_id in request.source_chunk_ids:
                    chunk = ChunkRepository(session).get(chunk_id)
                    if chunk is None:
                        raise HTTPException(
                            status_code=404,
                            detail=f"source chunk not found: {chunk_id}",
                        )
                    repository.add_source(topic, chunk)
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
            return topic_response(topic)

    @app.patch(
        "/api/v1/courses/{course_id}/topics/{topic_id}",
        response_model=TopicResponse,
        tags=["topics"],
    )
    def update_topic(
        course_id: uuid.UUID, topic_id: uuid.UUID, request: TopicUpdateRequest
    ) -> TopicResponse:
        with resolved_database.session() as session:
            repository = TopicRepository(session)
            topic = repository.get(topic_id)
            if topic is None or topic.course_id != course_id:
                raise HTTPException(status_code=404, detail="topic not found")
            parent: Topic | None | object = ...
            if "parent_id" in request.model_fields_set:
                parent = (
                    repository.get(request.parent_id) if request.parent_id else None
                )
                if request.parent_id and parent is None:
                    raise HTTPException(
                        status_code=404, detail="parent topic not found"
                    )
            title = request.title.strip() if request.title is not None else None
            if request.title is not None and not title:
                raise HTTPException(
                    status_code=422, detail="topic title cannot be blank"
                )
            try:
                repository.update(
                    topic,
                    title=title,
                    description=request.description.strip()
                    if request.description is not None
                    else None,
                    state=request.state,
                    parent=parent,
                    position=request.position,
                )
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
            return topic_response(topic)

    @app.delete(
        "/api/v1/courses/{course_id}/topics/{topic_id}",
        status_code=204,
        tags=["topics"],
    )
    def delete_topic(course_id: uuid.UUID, topic_id: uuid.UUID) -> Response:
        with resolved_database.session() as session:
            repository = TopicRepository(session)
            topic = repository.get(topic_id)
            if topic is None or topic.course_id != course_id:
                raise HTTPException(status_code=404, detail="topic not found")
            repository.delete(topic)
        return Response(status_code=204)

    def generation_job_response(job_id: uuid.UUID) -> GenerationJobResponse:
        with resolved_database.session() as session:
            job = GenerationJobRepository(session).get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="generation job not found")
            return GenerationJobResponse.model_validate(job)

    def enqueue_generation(
        *,
        course_id: uuid.UUID,
        kind: GenerationKind,
        idempotency_key: str | None,
        topic_id: uuid.UUID | None = None,
        item_id: uuid.UUID | None = None,
    ) -> GenerationJobResponse:
        with resolved_database.session() as session:
            job = GenerationJobRepository(session).add(
                GenerationJob(
                    course_id=course_id,
                    topic_id=topic_id,
                    item_id=item_id,
                    kind=kind,
                    status=GenerationJobStatus.QUEUED,
                    idempotency_key=idempotency_key or uuid.uuid4().hex,
                    prompt_version=PROMPT_VERSION,
                    model_name=resolved_chat_provider.model_name,
                )
            )
            job_id = job.id
            existing_status = job.status
        if (
            not app_settings.background_worker_enabled
            and existing_status is GenerationJobStatus.QUEUED
        ):
            for _ in range(app_settings.worker_max_attempts):
                generation_worker.run_job(job_id)
                result = generation_job_response(job_id)
                if result.status in {
                    GenerationJobStatus.COMPLETED,
                    GenerationJobStatus.FAILED,
                }:
                    return result
        return generation_job_response(job_id)

    @app.post(
        "/api/v1/courses/{course_id}/generate-curriculum",
        response_model=GenerationJobResponse,
        status_code=202,
        tags=["generation"],
    )
    def generate_curriculum(
        course_id: uuid.UUID, request: GenerationRequest
    ) -> GenerationJobResponse:
        with resolved_database.session() as session:
            if CourseRepository(session).get(course_id) is None:
                raise HTTPException(status_code=404, detail="course not found")
        return enqueue_generation(
            course_id=course_id,
            kind=GenerationKind.CURRICULUM,
            idempotency_key=request.idempotency_key,
        )

    @app.get(
        "/api/v1/generation-jobs/{job_id}",
        response_model=GenerationJobResponse,
        tags=["generation"],
    )
    def get_generation_job(job_id: uuid.UUID) -> GenerationJobResponse:
        return generation_job_response(job_id)

    @app.post(
        "/api/v1/courses/{course_id}/topics/{topic_id}/generate-items",
        response_model=GenerationJobResponse,
        status_code=202,
        tags=["generation"],
    )
    def generate_topic_items(
        course_id: uuid.UUID,
        topic_id: uuid.UUID,
        request: GenerationRequest,
    ) -> GenerationJobResponse:
        with resolved_database.session() as session:
            topic = TopicRepository(session).get(topic_id)
            if topic is None or topic.course_id != course_id:
                raise HTTPException(status_code=404, detail="topic not found")
        return enqueue_generation(
            course_id=course_id,
            topic_id=topic_id,
            kind=GenerationKind.STUDY_ITEMS,
            idempotency_key=request.idempotency_key,
        )

    def study_item_response(item: StudyItem) -> StudyItemResponse:
        return StudyItemResponse(
            id=item.id,
            topic_id=item.topic_id,
            objective_id=item.objective_id,
            item_type=item.item_type,
            prompt=item.prompt,
            answer=item.answer,
            explanation=item.explanation,
            options=item.options,
            difficulty=item.difficulty,
            approval_status=item.approval_status,
            generation_version=item.generation_version,
            sources=[
                SourcePassageResponse(
                    chunk_id=source.chunk.id,
                    document_id=source.chunk.document_id,
                    filename=source.chunk.document.filename,
                    position=source.chunk.position,
                    text=source.chunk.text,
                )
                for source in item.sources
            ],
            created_at=item.created_at,
            updated_at=item.updated_at,
        )

    @app.get(
        "/api/v1/courses/{course_id}/topics/{topic_id}/study-items",
        response_model=list[StudyItemResponse],
        tags=["study-items"],
    )
    def list_study_items(
        course_id: uuid.UUID, topic_id: uuid.UUID
    ) -> list[StudyItemResponse]:
        with resolved_database.session() as session:
            topic = TopicRepository(session).get(topic_id)
            if topic is None or topic.course_id != course_id:
                raise HTTPException(status_code=404, detail="topic not found")
            return [
                study_item_response(item)
                for item in StudyItemRepository(session).list_for_topic(topic_id)
            ]

    @app.patch(
        "/api/v1/study-items/{item_id}",
        response_model=StudyItemResponse,
        tags=["study-items"],
    )
    def update_study_item(
        item_id: uuid.UUID, request: StudyItemUpdateRequest
    ) -> StudyItemResponse:
        with resolved_database.session() as session:
            item = StudyItemRepository(session).get(item_id)
            if item is None:
                raise HTTPException(status_code=404, detail="study item not found")
            for field in ("prompt", "answer", "explanation"):
                value = getattr(request, field)
                if value is not None:
                    setattr(item, field, value.strip())
            if request.difficulty is not None:
                item.difficulty = request.difficulty
            if request.options is not None:
                options = [option.model_dump() for option in request.options]
                if (
                    item.item_type.value == "multiple_choice"
                    and sum(option["correct"] for option in options) != 1
                ):
                    raise HTTPException(
                        status_code=422,
                        detail="multiple-choice items require one correct option",
                    )
                item.options = options
            if request.approval_status is not None:
                item.approval_status = request.approval_status
                item.archived_at = (
                    datetime.now(UTC)
                    if request.approval_status is ApprovalStatus.ARCHIVED
                    else None
                )
            item.fingerprint = hashlib.sha256(
                " ".join(item.prompt.casefold().split()).encode()
            ).hexdigest()
            session.flush()
            return study_item_response(item)

    @app.delete("/api/v1/study-items/{item_id}", status_code=204, tags=["study-items"])
    def archive_study_item(item_id: uuid.UUID) -> Response:
        with resolved_database.session() as session:
            item = StudyItemRepository(session).get(item_id)
            if item is None:
                raise HTTPException(status_code=404, detail="study item not found")
            item.approval_status = ApprovalStatus.ARCHIVED
            item.archived_at = datetime.now(UTC)
        return Response(status_code=204)

    @app.post(
        "/api/v1/study-items/{item_id}/regenerate",
        response_model=GenerationJobResponse,
        status_code=202,
        tags=["generation"],
    )
    def regenerate_study_item(
        item_id: uuid.UUID, request: GenerationRequest
    ) -> GenerationJobResponse:
        with resolved_database.session() as session:
            item = StudyItemRepository(session).get(item_id)
            if item is None:
                raise HTTPException(status_code=404, detail="study item not found")
            course_id = item.topic.course_id
        return enqueue_generation(
            course_id=course_id,
            topic_id=item.topic_id,
            item_id=item.id,
            kind=GenerationKind.STUDY_ITEM,
            idempotency_key=request.idempotency_key,
        )

    @app.post(
        "/api/v1/courses/{course_id}/topics/{topic_id}/merge",
        response_model=TopicResponse,
        tags=["topics"],
    )
    def merge_topic(
        course_id: uuid.UUID, topic_id: uuid.UUID, request: TopicMergeRequest
    ) -> TopicResponse:
        with resolved_database.session() as session:
            repository = TopicRepository(session)
            source = repository.get(topic_id)
            target = repository.get(request.target_topic_id)
            if (
                source is None
                or target is None
                or source.course_id != course_id
                or target.course_id != course_id
                or source.id == target.id
            ):
                raise HTTPException(status_code=422, detail="merge topics are invalid")
            if source.state is TopicState.APPROVED:
                raise HTTPException(
                    status_code=409, detail="approved topics cannot be merged"
                )
            known_chunks = {link.chunk_id for link in target.sources}
            for link in source.sources:
                if link.chunk_id not in known_chunks:
                    repository.add_source(target, link.chunk)
            for objective in source.objectives:
                objective.topic = target
                objective.position = len(target.objectives)
            for item in source.study_items:
                item.topic = target
            repository.delete(source)
            session.flush()
            return topic_response(target)

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
                detail=(f"file exceeds the {app_settings.max_upload_bytes}-byte limit"),
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
            retrieval_cache.invalidate_retrieval(
                reason="document_deleted",
                session=session,
            )
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
    async def search_chunks(
        request: SearchRequest,
        response: Response,
    ) -> SearchResponse:
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
        response.headers["X-Embedding-Cache"] = result.embedding_cache_status
        response.headers["X-Retrieval-Cache"] = result.retrieval_cache_status
        response.headers["X-Corpus-Version"] = str(result.corpus_version)
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
                candidate_k=(request.candidate_k or app_settings.context_candidate_k),
                max_chunks=(request.max_chunks or app_settings.context_max_chunks),
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
                    ChatMessageResponse.model_validate(message) for message in messages
                ],
            )

    frontend_dist = app_settings.frontend_dist_path
    if not frontend_dist.is_absolute():
        frontend_dist = Path.cwd() / frontend_dist
    if not frontend_dist.is_dir():
        frontend_dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
    if frontend_dist.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=frontend_dist, html=True),
            name="operator-interface",
        )

    return app


def _check_database(database: Database) -> None:
    with database.engine.connect() as connection:
        connection.execute(text("SELECT 1"))


app = create_app()
