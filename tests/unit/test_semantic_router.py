from __future__ import annotations

from rag_ops_guard.agent.router import Route, SemanticRouter


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        normalized = text.casefold()
        if "hola" in normalized:
            return [1.0, 0.0, 0.0, 0.0]
        if "document" in normalized:
            return [0.0, 1.0, 0.0, 0.0]
        if "api" in normalized:
            return [0.0, 0.0, 1.0, 0.0]
        if "weather" in normalized:
            return [0.0, 0.0, 0.0, 1.0]
        return [0.5, 0.5, 0.5, 0.5]


def _examples() -> dict[Route, list[str]]:
    return {
        "chat": ["hola"],
        "capabilities": ["help capabilities"],
        "catalog": ["documentation"],
        "knowledge": ["api"],
        "out_of_scope": ["weather"],
        "uncertain": [],
    }


def test_router_uses_best_individual_example_for_catalog() -> None:
    router = SemanticRouter(FakeEmbeddings(), route_examples=_examples())

    decision = router.route("What documentation is available?")

    assert decision.route == "catalog"
    assert decision.score == 1.0


def test_router_sends_operational_question_to_knowledge() -> None:
    router = SemanticRouter(FakeEmbeddings(), route_examples=_examples())

    decision = router.route("What does this API do?")

    assert decision.route == "knowledge"


def test_router_can_abstain_when_routes_are_too_close() -> None:
    router = SemanticRouter(
        FakeEmbeddings(),
        route_examples=_examples(),
        min_score=0.0,
        min_margin=0.1,
    )

    decision = router.route("ambiguous question")

    assert decision.route == "uncertain"
    assert decision.margin == 0.0
