from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager

from rag_ops_guard.domain.models import QueryContext

_CURRENT_QUERY_CONTEXT: contextvars.ContextVar[QueryContext | None] = contextvars.ContextVar(
    "rag_ops_query_context",
    default=None,
)


def current_query_context() -> QueryContext:
    """Return the query context bound to the current tool execution."""

    return _CURRENT_QUERY_CONTEXT.get() or QueryContext()


@contextmanager
def bind_query_context(context: QueryContext) -> Iterator[None]:
    """Bind request-scoped context while a tool executes."""

    token = _CURRENT_QUERY_CONTEXT.set(context)
    try:
        yield
    finally:
        _CURRENT_QUERY_CONTEXT.reset(token)
