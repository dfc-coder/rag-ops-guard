class RagOpsError(Exception):
    """Base domain error."""


class DocumentValidationError(RagOpsError):
    """Raised when a knowledge document is invalid."""


class CitationValidationError(RagOpsError):
    """Raised when generated citations are not grounded in resolved evidence."""
