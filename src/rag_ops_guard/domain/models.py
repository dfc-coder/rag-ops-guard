from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DocumentStatus(StrEnum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    DRAFT = "draft"


class DocumentType(StrEnum):
    RUNBOOK = "runbook"
    API = "api"
    SLA = "sla"
    INCIDENT = "incident"
    POSTMORTEM = "postmortem"
    ARCHITECTURE = "architecture"


class QueryStatus(StrEnum):
    ANSWERED = "answered"
    ANSWERED_UNGROUNDED = "answered_ungrounded"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CLARIFICATION_REQUIRED = "clarification_required"
    SAFETY_BLOCKED = "safety_blocked"
    ERROR = "error"


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    logical_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    version: str = Field(min_length=1)
    status: DocumentStatus
    effective_date: date
    system: str = Field(min_length=1)
    environment: Literal["production", "staging", "all"]
    document_type: DocumentType
    authority: int = Field(ge=0, le=100)
    supersedes: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    logical_id: str
    version: str
    title: str
    header_path: list[str] = Field(default_factory=list)
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    metadata: DocumentMetadata
    content_sha256: str


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk: Chunk
    distance: float | None = None


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    logical_id: str
    title: str
    version: str
    chunk_id: str
    s3_key: str


class QueryContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system: str | None = None
    environment: Literal["production", "staging"] | None = None
    api_version: str | None = None


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    context: QueryContext = Field(default_factory=QueryContext)
    thread_id: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("question", mode="before")
    @classmethod
    def normalize_question(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must contain non-whitespace text")
        return normalized


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    status: QueryStatus
    route: (
        Literal[
            "chat",
            "capabilities",
            "catalog",
            "knowledge",
            "out_of_scope",
            "uncertain",
            "safety",
            "error",
        ]
        | None
    ) = None
    answer: str | None = None
    clarification_question: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    route_confidence: float | None = None
    route_margin: float | None = None
    relevance_score: float | None = None
    retrieval_query: str | None = None
    rewritten_query: str | None = None

    @model_validator(mode="after")
    def validate_status_contract(self) -> QueryResponse:
        answered_statuses = {QueryStatus.ANSWERED, QueryStatus.ANSWERED_UNGROUNDED}
        if self.status in answered_statuses and not self.answer:
            raise ValueError("answered responses require an answer")

        if self.status == QueryStatus.ANSWERED_UNGROUNDED and self.citations:
            raise ValueError("answered_ungrounded responses cannot carry citations")

        if self.route not in {"knowledge", None} and self.citations:
            raise ValueError("only knowledge responses may carry citations")

        if (
            self.status == QueryStatus.ANSWERED
            and self.route in {"knowledge", None}
            and not self.citations
        ):
            raise ValueError("grounded answered responses require citations")

        if self.status not in answered_statuses and self.citations:
            raise ValueError("non-answered responses cannot carry citations")

        if self.status == QueryStatus.CLARIFICATION_REQUIRED and not self.clarification_question:
            raise ValueError("clarification_required requires clarification_question")
        return self


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    s3_key: str = Field(min_length=1)


class IngestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ingested", "no_op"]
    document_id: str
    logical_id: str
    version: str
    chunks: int = Field(ge=0)


class QueryAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_question: str = Field(min_length=1)
    systems: list[str] = Field(default_factory=list)
    environment: Literal["production", "staging"] | None = None
    api_version: str | None = None
    requires_clarification: bool
    clarification_question: str | None = None
    safety_category: Literal["normal", "secret_extraction", "policy_bypass"]
    fallback_message: str | None = Field(default=None, min_length=1, max_length=180)
    safety_blocked_message: str | None = Field(default=None, min_length=1, max_length=180)
    insufficient_evidence_message: str | None = Field(default=None, min_length=1, max_length=180)

    @model_validator(mode="after")
    def normalize_fallback_messages(self) -> QueryAnalysis:
        fallback = (
            self.fallback_message
            or self.insufficient_evidence_message
            or self.safety_blocked_message
            or "The available documentation does not provide enough evidence to answer safely."
        )
        self.fallback_message = fallback
        self.safety_blocked_message = self.safety_blocked_message or fallback
        self.insufficient_evidence_message = self.insufficient_evidence_message or fallback
        return self


class GroundedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "answered",
        "insufficient_evidence",
        "clarification_required",
        "safety_blocked",
    ]
    answer: str = Field(min_length=1)
    citation_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_grounding_contract(self) -> GroundedAnswer:
        normalized = self.answer.strip().lower()
        looks_like_backend_timeout = "timed out" in normalized and "shorter message" in normalized

        if self.status == "answered" and (not self.citation_ids or looks_like_backend_timeout):
            self.status = "insufficient_evidence"
            self.citation_ids = []

        if self.status != "answered":
            self.citation_ids = []

        return self


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    logical_id: str
    version: str
    content_sha256: str
    vector_keys: list[str]
