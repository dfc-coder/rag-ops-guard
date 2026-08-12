from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CLARIFICATION_REQUIRED = "clarification_required"
    SAFETY_BLOCKED = "safety_blocked"


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

    question: str = Field(min_length=3, max_length=2000)
    context: QueryContext = Field(default_factory=QueryContext)


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    status: QueryStatus
    answer: str | None = None
    clarification_question: str | None = None
    citations: list[Citation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status_contract(self) -> QueryResponse:
        if self.status == QueryStatus.ANSWERED:
            if not self.answer or not self.citations:
                raise ValueError("answered responses require answer and citations")
        if self.status == QueryStatus.CLARIFICATION_REQUIRED:
            if not self.clarification_question:
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


class GroundedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["answered", "insufficient_evidence"]
    answer: str
    citation_ids: list[str] = Field(default_factory=list)


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    logical_id: str
    version: str
    content_sha256: str
    vector_keys: list[str]
