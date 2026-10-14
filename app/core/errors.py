"""The one place that names every failure: code, HTTP status, and default public message.

Adapters raise by code; nothing else in the codebase repeats a status or a message.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Failure:
    status: int
    message: str


FAILURES: dict[str, Failure] = {
    # model provider
    "provider_timeout": Failure(504, "The local routing model timed out."),
    "provider_unavailable": Failure(503, "The local routing model is unavailable."),
    "provider_invalid_response": Failure(502, "The local routing model rejected the request."),
    "provider_invalid_output": Failure(502, "The local routing model returned invalid routing output."),
    "model_unavailable": Failure(503, "The configured local routing model is unavailable."),
    # request handling
    "context_limit": Failure(413, "This conversation is too long to route; start a new focused conversation."),
    "database_unavailable": Failure(503, "The local database is unavailable."),
    "interrupted": Failure(503, "Routing was interrupted before it completed."),
    "routing_failed": Failure(503, "The original routing request failed."),
    # conflicts and lookups
    "conversation_busy": Failure(409, "This conversation already has a routing request in progress."),
    "message_conflict": Failure(409, "This message ID is already associated with different content or scope."),
    "import_conflict": Failure(409, "An imported conversation conflicts with existing saved history."),
    "manual_scope_conflict": Failure(409, "A selected manual is unavailable."),
    "conversation_not_found": Failure(404, "Conversation not found."),
    "routing_run_not_found": Failure(404, "Routing run not found."),
}

_UNKNOWN = FAILURES["routing_failed"]


def failure_for(code: str | None) -> Failure:
    """Look up a code; a code stored by an older build falls back to the generic failure."""
    return FAILURES.get(code or "", _UNKNOWN)


def status_for(code: str | None) -> int:
    return failure_for(code).status


class ServiceError(Exception):
    """A typed failure carrying the code, a public message, and optional context for the response body."""

    def __init__(self, code: str, message: str | None = None, **fields: Any) -> None:
        self.code = code
        self.message = message or failure_for(code).message
        self.status_code = status_for(code)
        self.fields = fields
        super().__init__(self.message)

    def body(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, **{k: v for k, v in self.fields.items() if v is not None}}
