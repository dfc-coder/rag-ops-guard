class RagOpsError(Exception):
    """Base domain error."""


class DocumentValidationError(RagOpsError):
    """Raised when a knowledge document is invalid."""


class CitationValidationError(RagOpsError):
    """Raised when generated citations are not grounded in resolved evidence."""


class EvidenceConflictError(RagOpsError):
    """Raised when equally authoritative active documents cannot be resolved safely."""


class ModelTimeoutError(RagOpsError):
    """Raised when the local generation model exceeds its request timeout."""


class InvalidModelResponseError(RagOpsError):
    """Raised when the model returns a backend/error message as an answer."""
