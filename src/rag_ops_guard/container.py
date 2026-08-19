from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import Generic, TypeVar

from rag_ops_guard.tenancy import RequestContext

T = TypeVar("T")
Factory = Callable[[RequestContext, str], T]


class Container(Generic[T]):
    """Bounded LRU cache keyed by authenticated tenant and immutable config generation."""

    def __init__(self, *, max_generations: int = 32) -> None:
        if max_generations < 1:
            raise ValueError("max_generations must be positive")
        self._max_generations = max_generations
        self._entries: OrderedDict[tuple[str, str], T] = OrderedDict()

    def get(self, context: RequestContext, config_hash: str, factory: Factory[T]) -> T:
        key = (context.tenant_id, config_hash)
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
            return cached

        value = factory(context, config_hash)
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_generations:
            self._entries.popitem(last=False)
        return value

    def clear(self) -> None:
        self._entries.clear()

    def keys(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._entries)

    def __len__(self) -> int:
        return len(self._entries)
