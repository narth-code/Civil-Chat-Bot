"""Maps typed service failures and database outages to JSON error responses."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.core.errors import ServiceError

logger = logging.getLogger(__name__)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def service_error(_: Request, error: ServiceError) -> JSONResponse:
        return JSONResponse(status_code=error.status_code, content=jsonable_encoder(error.body()))

    @app.exception_handler(SQLAlchemyError)
    @app.exception_handler(OSError)
    async def database_error(_: Request, error: Exception) -> JSONResponse:
        logger.error("database operation failed: %s", type(error).__name__, exc_info=error)
        failure = ServiceError("database_unavailable")
        return JSONResponse(status_code=failure.status_code, content=failure.body())
