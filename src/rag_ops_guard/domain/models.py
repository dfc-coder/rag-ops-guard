from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)


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
    ANSWERED_GROUNDED = "answered_grounded"
    ANSWERED_MIXED = "answered_mixed"
    ANSWERED_UNGROUNDED = "answered_ungrounded"
    CLARIFICATION_REQUIRED = "clarification_required"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    SAFETY_BLOCKED = "safety_blocked"
    ERROR = "error"


class ResponseOutcome(StrEnum):
    ANSWER = "answer"
    CLARIFICATION_REQUIRED = "clarification_required"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    SAFETY_BLOCKED = "safety_blocked"
    ERROR = "error"


class DocumentMetadata(BaseModel):
    """Document identity plus optional governance metadata."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    logical_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    version: str = Field(min_length=1)
    status: DocumentStatus | None = None
    effective_date: date | None = None
    system: str | None = Field(default=None, min_length=1)
    environment: Literal["production", "staging", "all"] | None = None
    document_type: DocumentType | None = None
    authority: int | None = Field(default=None, ge=0, le=100)
    supersedes: list[str] = Field(default_factory=list)

    @property
    def has_governance_metadata(self) -> bool:
        return any(
            (
                self.status is not None,
                self.effective_date is not None,
                self.system is not None,
                self.environment is not None,
                self.document_type is not None,
                self.authority is not None,
                bool(self.supersedes),
            )
        )


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


class GeneratedSegment(BaseModel):
    """Model-produced segment before citation IDs are validated/materialized."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    citation_ids: list[str] = Field(default_factory=list)

    @field_validator("text", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("citation_ids")
    @classmethod
    def dedupe_citation_ids(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(item.strip() for item in value if item.strip()))


class StructuredAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segments: list[GeneratedSegment] = Field(min_length=1)


class ResponseSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    citations: list[Citation] = Field(default_factory=list)

    @field_validator("text", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @property
    def grounded(self) -> bool:
        return bool(self.citations)

    @model_serializer(mode="wrap")
    def serialize_with_grounding(self, handler: Any) -> dict[str, Any]:
        data = dict(handler(self))
        data["grounded"] = self.grounded
        return data


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
    """Public response contract; answer status/citations are derived from validated segments."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    outcome: ResponseOutcome = ResponseOutcome.ANSWER
    segments: list[ResponseSegment] = Field(default_factory=list)
    message: str | None = None
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
    clarification_question: str | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
    route_confidence: float | None = None
    route_margin: float | None = None
    relevance_score: float | None = None
    domain_relevance_score: float | None = None
    grounded_relevance_score: float | None = None
    retrieval_query: str | None = None
    rewritten_query: str | None = None
    config_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_outcome_contract(self) -> QueryResponse:
        if self.outcome == ResponseOutcome.ANSWER and not self.segments:
            raise ValueError("answer responses require at least one segment")
        if self.outcome != ResponseOutcome.ANSWER and self.segments:
            raise ValueError("non-answer responses cannot carry answer segments")
        if (
            self.outcome == ResponseOutcome.CLARIFICATION_REQUIRED
            and not self.clarification_question
        ):
            raise ValueError("clarification_required requires clarification_question")
        return self

    @property
    def status(self) -> QueryStatus:
        if self.outcome == ResponseOutcome.CLARIFICATION_REQUIRED:
            return QueryStatus.CLARIFICATION_REQUIRED
        if self.outcome == ResponseOutcome.INSUFFICIENT_EVIDENCE:
            return QueryStatus.INSUFFICIENT_EVIDENCE
        if self.outcome == ResponseOutcome.SAFETY_BLOCKED:
            return QueryStatus.SAFETY_BLOCKED
        if self.outcome == ResponseOutcome.ERROR:
            return QueryStatus.ERROR

        grounded = sum(1 for segment in self.segments if segment.grounded)
        if grounded == len(self.segments):
            return QueryStatus.ANSWERED_GROUNDED
        if grounded:
            return QueryStatus.ANSWERED_MIXED
        return QueryStatus.ANSWERED_UNGROUNDED

    @property
    def answer(self) -> str | None:
        if self.outcome == ResponseOutcome.ANSWER:
            return "\n\n".join(segment.text for segment in self.segments)
        return self.message

    @property
    def citations(self) -> list[Citation]:
        unique: dict[str, Citation] = {}
        for segment in self.segments:
            for citation in segment.citations:
                unique.setdefault(citation.chunk_id, citation)
        return list(unique.values())

    @model_serializer(mode="wrap")
    def serialize_with_derived_contract(self, handler: Any) -> dict[str, Any]:
        data = dict(handler(self))
        data["status"] = self.status.value
        data["answer"] = self.answer
        data["citations"] = [citation.model_dump(mode="json") for citation in self.citations]
        return data


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


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    logical_id: str
    version: str
    content_sha256: str
    vector_keys: list[str]
