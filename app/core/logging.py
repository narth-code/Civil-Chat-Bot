"""Process-wide console and rotating API file logging."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.config import Settings

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_HANDLER_MARKER = "_civil_chatbot_api_handler"
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "sentence_transformers",
    "huggingface_hub",
    "docling",
    "RapidOCR",
    "rapidocr",
    "PIL",
)


def configure_logging(settings: Settings) -> None:
    """Install this application's handlers once, preserving other logging integrations."""
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "api.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter(_FORMAT)

    existing = {
        getattr(handler, _HANDLER_MARKER): handler for handler in root.handlers if hasattr(handler, _HANDLER_MARKER)
    }
    if "file" not in existing:
        file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
        file_handler.setFormatter(formatter)
        setattr(file_handler, _HANDLER_MARKER, "file")
        root.addHandler(file_handler)
    if "console" not in existing:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        setattr(console_handler, _HANDLER_MARKER, "console")
        root.addHandler(console_handler)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    for name in logging.root.manager.loggerDict:
        if name.startswith("docling."):
            logging.getLogger(name).setLevel(logging.WARNING)
