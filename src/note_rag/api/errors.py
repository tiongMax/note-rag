"""Consistent API error responses."""

import logging
from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    details: Any = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    payload: dict[str, Any] = {
        "error": {
            "code": code,
            "message": message,
            "request_id": getattr(request.state, "request_id", None),
        }
    }
    if details is not None:
        payload["error"]["details"] = details
    return JSONResponse(
        status_code=status_code,
        content=payload,
        headers=headers,
    )


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request,
        error: HTTPException,
    ) -> JSONResponse:
        message = (
            error.detail
            if isinstance(error.detail, str)
            else "The request could not be completed."
        )
        details = None if isinstance(error.detail, str) else error.detail
        return error_response(
            request,
            status_code=error.status_code,
            code=f"http_{error.status_code}",
            message=message,
            details=details,
            headers=error.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        return error_response(
            request,
            status_code=422,
            code="validation_error",
            message="Request validation failed.",
            details=error.errors(),
        )

    @app.exception_handler(Exception)
    async def unexpected_exception_handler(
        request: Request,
        error: Exception,
    ) -> JSONResponse:
        logger.exception(
            "Unhandled request error",
            extra={"request_id": getattr(request.state, "request_id", None)},
        )
        return error_response(
            request,
            status_code=500,
            code="internal_error",
            message="An unexpected server error occurred.",
        )
