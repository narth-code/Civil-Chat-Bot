"""Strict transport contracts shared by the local API and UI."""

from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.manuals import manual_ids


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)


class Route(str, Enum):
    retrieval = "retrieval"
    conversation = "conversation"
    invalid = "invalid"


class ReasonCode(str, Enum):
    code_lookup = "code_lookup"
    greeting = "greeting"
    clarification = "clarification"
    workspace_help = "workspace_help"
    unsafe_request = "unsafe_request"
    prompt_injection = "prompt_injection"
    off_topic = "off_topic"


ROUTE_OF_REASON = {
    ReasonCode.code_lookup: Route.retrieval,
    ReasonCode.greeting: Route.conversation,
    ReasonCode.clarification: Route.conversation,
    ReasonCode.workspace_help: Route.conversation,
    ReasonCode.unsafe_request: Route.invalid,
    ReasonCode.prompt_injection: Route.invalid,
    ReasonCode.off_topic: Route.invalid,
}


class RouteDecision(Contract):
    route: Route
    reason_code: ReasonCode
    rationale: str = Field(min_length=1, max_length=400)

    @model_validator(mode="after")
    def valid_pair(self):
        if ROUTE_OF_REASON[self.reason_code] != self.route:
            raise ValueError("reason_code is incompatible with route")
        return self


ManualId = str


class ManualScope(Contract):
    selected_manual_ids: list[ManualId] = Field(default_factory=list)

    @field_validator("selected_manual_ids")
    @classmethod
    def unique_scope(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("selected_manual_ids must be unique")
        return value


class RequestedManualScope(ManualScope):
    """Accepted input: every id must be in the manifest. Output models use ManualScope and trust stored ids."""

    @field_validator("selected_manual_ids")
    @classmethod
    def known_scope(cls, value):
        unknown = sorted(set(value) - set(manual_ids()))
        if unknown:
            raise ValueError(f"unknown manual ids: {', '.join(unknown)}")
        return value


class ConversationCreate(Contract):
    pass


class ConversationSummary(Contract):
    id: UUID
    title: str
    created_at: AwareDatetime
    updated_at: AwareDatetime


class ConversationList(Contract):
    conversations: list[ConversationSummary]


class Message(ManualScope):
    id: UUID
    conversation_id: UUID
    sequence: int = Field(ge=0)
    role: Literal["user", "assistant"]
    content: str
    created_at: AwareDatetime
    origin: Literal["user", "router_status", "assistant_answer", "legacy_demo"]
    routing_run_id: UUID | None


class MessageCreate(RequestedManualScope):
    message_id: UUID
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def nonblank_content(cls, value):
        if not value.strip():
            raise ValueError("content must not be blank")
        return value


class ImportedMessage(RequestedManualScope):
    id: UUID
    role: Literal["user", "assistant"]
    content: str
    created_at: AwareDatetime


class ImportedConversation(Contract):
    id: UUID
    title: str
    created_at: AwareDatetime
    messages: list[ImportedMessage]


class ConversationImport(Contract):
    conversations: list[ImportedConversation]

    @model_validator(mode="after")
    def unique_ids(self):
        conversations = [c.id for c in self.conversations]
        messages = [m.id for c in self.conversations for m in c.messages]
        if len(set(conversations)) != len(conversations) or len(set(messages)) != len(messages):
            raise ValueError("import identities must be unique")
        return self


class ImportResult(Contract):
    imported_conversation_ids: list[UUID]


class EvaluationCreate(Contract):
    expected_route: Route
    comment: str = Field(default="", max_length=5000)


class Evaluation(EvaluationCreate):
    id: UUID
    run_id: UUID
    created_at: AwareDatetime


class Citation(Contract):
    type: Literal["provision", "web"]
    document_id: str | None = None
    identifier: str | None = None
    title: str | None = None
    pages: str | None = None
    quote: str | None = None
    url: str | None = None


class RoutingRunView(Contract):
    id: UUID
    conversation_id: UUID
    user_message_id: UUID
    question: str
    status: Literal["pending", "classified", "failed"]
    decision: RouteDecision | None
    outcome: str | None
    citations: list[Citation]
    review: dict[str, Any] | None
    trace: list[str]
    city: str | None
    error_code: str | None
    created_at: AwareDatetime
    completed_at: AwareDatetime | None
    model: str
    model_digest: str | None
    policy_version: str
    model_options: dict[str, Any]
    elapsed_ms: float | None
    routing_ms: float | None
    model_calls: int = Field(ge=0, le=1)
    context_message_ids: list[UUID]
    selected_manual_ids: list[ManualId]
    selected_source_version_ids: list[UUID]
    latest_evaluation: Evaluation | None


class ConversationDetail(Contract):
    conversation: ConversationSummary
    messages: list[Message]
    pending_run: RoutingRunView | None


class MessageResult(Contract):
    messages: list[Message]
    routing_run: RoutingRunView


class ErrorPayload(Contract):
    code: str
    message: str
    conversation_id: UUID | None = None
    message_id: UUID | None = None
    run_id: UUID | None = None
    messages: list[Message] | None = None
    routing_run: RoutingRunView | None = None


class RoutingWindow(Contract):
    days: Literal[7, 30, 90]
    start: AwareDatetime
    end: AwareDatetime


class LatencyMetrics(Contract):
    sample_count: int = Field(ge=0)
    median_ms: float | None
    p95_ms: float | None


class EvaluationMetrics(Contract):
    reviewed_count: int = Field(ge=0)
    agreement_count: int = Field(ge=0)
    agreement_rate: float | None = Field(ge=0, le=1)


class RoutingMetrics(Contract):
    total: int
    pending: int
    failed: int
    classified: int
    route_counts: dict[Route, int]
    invalid_reason_counts: dict[str, int]
    latency: LatencyMetrics
    evaluation: EvaluationMetrics
    outcome_counts: dict[str, int]


class RoutingList(Contract):
    window: RoutingWindow
    metrics: RoutingMetrics
    items: list[RoutingRunView]
    total: int
    offset: int
    limit: int


class PolicyView(Contract):
    version: str
    system_prompt: str
    response_schema: dict[str, Any]


class TraceEvent(Contract):
    seq: int = Field(default=0, ge=0)
    node: str
    kind: Literal["node", "llm_call", "retrieval", "web_search", "decision"]
    started_at: AwareDatetime
    duration_ms: int = Field(ge=0)
    summary: str
    data: dict[str, Any]


class RoutingRunTrace(Contract):
    run_id: UUID
    events: list[TraceEvent]


class RoutingDetail(Contract):
    routing_run: RoutingRunView
    context_messages: list[Message]
    policy: PolicyView
    evaluations: list[Evaluation]


class ManualView(Contract):
    id: ManualId
    label: str
    title: str
    filename: str
    url: str
    current_source_version_id: UUID | None
    edition: str | None
    jurisdiction: str


class ManualCatalog(Contract):
    manuals: list[ManualView]
