"""State and structured-output schemas of the agentic RAG graph (app/rag/agent.py)."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, create_model

from app.schemas import ReasonCode


class AgentState(TypedDict):
    question: str
    history: list[dict]
    current_query: str
    city: str | None
    decision: dict | None
    manual_hint: str | None
    scope: list[str] | None
    sub_location: str | None
    section_ref: str | None
    section_missing: bool
    edition_gap: bool
    kb_docs: list[Any]
    web_results: str
    kb_grade: str
    required_facts: list[str]
    web_grade: str
    regimes: list[str]
    clarification: str | None
    answer: str
    source_used: str
    retry_count: int
    kb_retried: bool
    grader_labels: list[str]
    trace: list[str]
    events: list[dict[str, Any]]
    citations: list[dict]
    quote_check: dict | None
    review: dict | None
    failure: str | None


class FactEvidence(BaseModel):
    """A requested fact and the evidence labels (c<i>.s<j>) that state it; empty when no evidence states it."""

    fact: str
    labels: list[str] = Field(default_factory=list)


class EvidenceGrade(BaseModel):
    """KB fact grading: the model maps each requested fact to evidence labels; the verdict is computed in code."""

    reasoning: str
    facts: list[FactEvidence] = Field(default_factory=list)


class RegimeCheck(BaseModel):
    """Yes/no answer on one Evidence line that carves an exception out of a broader rule; the answer is constrained."""

    reasoning: str
    applies: Literal["yes", "no"]


def constrain_labels(schema: type[BaseModel], usable: list[str]) -> type[BaseModel]:
    """The schema with each evidence-label field limited to the usable tokens, so decoding cannot emit other text.

    With no usable token the schema is returned unchanged; `usable_labels` still filters every answer in code.
    """
    if not usable:
        return schema
    token = Literal[tuple(usable)]
    if schema is EvidenceGrade:
        fact = create_model(
            "FactEvidenceChoice", __base__=FactEvidence, labels=(list[token], Field(default_factory=list))
        )
        return create_model(
            "EvidenceGradeChoice", __base__=EvidenceGrade, facts=(list[fact], Field(default_factory=list))
        )
    raise ValueError(f"no label-constrained schema for {schema.__name__}")


class WebGrade(BaseModel):
    reasoning: str
    grade: Literal["good", "weak"] = "weak"


class ContextExtraction(BaseModel):
    city: str | None = None
    manual_hint: str | None = None
    sub_location: str | None = None
    section_ref: str | None = None
    search_query: str


class CityExtraction(BaseModel):
    city: str | None


class RouteReason(BaseModel):
    reason_code: ReasonCode
    rationale: str = Field(min_length=1, max_length=400)


class AnswerWithQuotes(BaseModel):
    answer: str
    supporting: list[str] = Field(default_factory=list)


class AnswerReview(BaseModel):
    valid: bool
    confidence: float = Field(ge=0, le=1)
    reasoning: str
