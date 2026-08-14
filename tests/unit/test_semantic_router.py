from __future__ import annotations

from rag_ops_guard.agent.router import SemanticRouter


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        normalized = text.casefold()
        if any(term in normalized for term in ("hola", "hello", "qué haces", "who are you")):
            return [1.0, 0.0]
        return [0.0, 1.0]


def test_router_sends_meta_conversation_to_chat() -> None:
    router = SemanticRouter(
        FakeEmbeddings(),
        chat_examples=["hola", "qué haces"],
        knowledge_examples=["qué dice la documentación", "qué pasó en el incidente"],
    )

    assert router.route("Hola, ¿qué haces?") == "chat"


def test_router_sends_operational_question_to_knowledge() -> None:
    router = SemanticRouter(
        FakeEmbeddings(),
        chat_examples=["hola", "qué haces"],
        knowledge_examples=["qué dice la documentación", "qué pasó en el incidente"],
    )

    assert router.route("¿Qué dice la documentación sobre el servicio?") == "knowledge"
