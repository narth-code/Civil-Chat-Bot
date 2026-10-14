from __future__ import annotations

import uvicorn

from app.core.config import get_settings
from app.core.logging import configure_logging


if __name__ == "__main__":
    settings = get_settings()
    configure_logging(settings)
    uvicorn.run("app.main:app", host=settings.backend_host, port=settings.backend_port, workers=1, log_config=None)
