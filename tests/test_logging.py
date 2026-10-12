"""API logging writes one file copy per record without storing request bodies."""

import logging
from logging.handlers import RotatingFileHandler

import httpx
import pytest

from app.core.config import Settings
from app.core.logging import configure_logging
from app.main import create_app


@pytest.mark.asyncio
async def test_api_logging_is_idempotent_and_omits_request_body(tmp_path):
    root = logging.getLogger()
    previous_handlers = [handler for handler in root.handlers if hasattr(handler, "_civil_chatbot_api_handler")]
    for handler in previous_handlers:
        root.removeHandler(handler)
    try:
        settings = Settings(log_dir=str(tmp_path))
        configure_logging(settings)
        configure_logging(settings)
        app = create_app(settings)
        file_handlers = [
            handler
            for handler in root.handlers
            if isinstance(handler, RotatingFileHandler)
            and getattr(handler, "_civil_chatbot_api_handler", None) == "file"
        ]
        assert len(file_handlers) == 1
        assert file_handlers[0].maxBytes == 5 * 1024 * 1024
        assert file_handlers[0].backupCount == 5
        logging.getLogger("test.logging").info("file record reached disk")

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/not-a-route", json={"content": "private-message-body"})
        assert response.status_code == 404
        file_handlers[0].flush()
        logged = (tmp_path / "api.log").read_text(encoding="utf-8")
        assert logged.count("file record reached disk") == 1
        assert "POST /api/not-a-route 404 " in logged
        assert "private-message-body" not in logged
    finally:
        for handler in list(root.handlers):
            if hasattr(handler, "_civil_chatbot_api_handler"):
                root.removeHandler(handler)
                if handler not in previous_handlers:
                    handler.close()
        for handler in previous_handlers:
            root.addHandler(handler)
