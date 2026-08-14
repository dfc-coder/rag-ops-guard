from __future__ import annotations

from rag_ops_guard.agent.router import DEFAULT_ROUTE_EXAMPLES, Route, SemanticRouter


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


class MappingEmbeddings:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectors[text]


def _examples() -> dict[Route, list[str]]:
    return {
        "chat": ["hola"],
        "catalog": ["documentation"],
        "knowledge": ["api"],
        "out_of_scope": ["weather"],
        "uncertain": [],
    }


def test_capability_examples_cover_direct_role_question() -> None:
    assert "¿Qué haces?" in DEFAULT_ROUTE_EXAMPLES["capabilities"]


def test_router_routes_clear_catalog_question() -> None:
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


def test_router_averages_prototypes_instead_of_following_one_collision() -> None:
    embeddings = MappingEmbeddings(
        {
            "query": [1.0, 0.0],
            "cap-1": [0.8, 0.6],
            "cap-2": [0.8, -0.6],
            "knowledge-collision": [0.99, 0.141067],
            "knowledge-other": [0.0, 1.0],
            "catalog-1": [0.6, 0.8],
            "catalog-2": [0.5, -0.866025],
        }
    )
    router = SemanticRouter(
        embeddings,  # type: ignore[arg-type]
        route_examples={
            "capabilities": ["cap-1", "cap-2"],
            "knowledge": ["knowledge-collision", "knowledge-other"],
            "catalog": ["catalog-1", "catalog-2"],
        },
        min_score=0.0,
        min_margin=0.0,
    )

    decision = router.route("query")

    assert decision.route == "capabilities"
    assert decision.scores["capabilities"] > decision.scores["knowledge"]
