"""Command-line entry point for the FastAPI application."""

import logging
import threading

import uvicorn

from note_rag.api.app import create_app

logger = logging.getLogger(__name__)


def main_api() -> int:
    """Run the FastAPI web server."""
    uvicorn.run(
        "note_rag.api.app:app",
        host="127.0.0.1",
        port=8001,
        reload=False,
    )
    return 0


def main_worker() -> int:
    """Run only the background ingestion worker loop (no web server)."""
    # Create the app to wire up dependencies, but don't start uvicorn
    app = create_app()
    worker = app.state.ingestion_worker
    
    logger.info("Starting IngestionWorker in standalone mode...")
    stop_event = threading.Event()
    try:
        worker.run_forever(stop_event)
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, stopping worker...")
        stop_event.set()
    finally:
        # Important: shut down the DB connection pool cleanly
        app.state.database.dispose()
        
    return 0
