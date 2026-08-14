from __future__ import annotations

from langchain_core.messages import BaseMessage

from rag_ops_guard.domain.models import GroundedAnswer, QueryRequest
from rag_ops_guard.graph.conversational_agent import ConversationalAgent
from rag_ops_guard.retrieval.hybrid import KnowledgeSearchResult
from tests.fixtures.builders import evidence


class FakeRouter:
    def route(self, text: str) -> str:
        return "chat" if "qué haces" in text.casefold() else "knowledge"


class FakeKnowledge:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.refreshed = False

    def search(self, query: str, context: object) -> KnowledgeSearchResult:
        del context
        self.queries.append(query)
        item = evidence(text="After the third retry, escalate to Treasury Integrations.")
        return KnowledgeSearchResult(
            dense=[item],
            lexical=[item],
            fused=[item],
            admitted=[item],
        )

    def refresh(self) -> None:
        self.refreshed = True


class FakeChat:
    def __init__(self) -> None:
        self.chat_calls = 0
        self.answer_calls = 0
        self.histories: list[list[BaseMessage]] = []

    def analyze_query(self, prompt: str) -> object:
        raise AssertionError(prompt)

    def generate_chat(self, messages: list[BaseMessage]) -> str:
        self.chat_calls += 1
        self.histories.append(messages)
        return "Soy RAG Ops Guard."

    def generate_answer(
        self,
        prompt: str,
        history: list[BaseMessage] | None = None,
    ) -> GroundedAnswer:
        del prompt
        self.answer_calls += 1
        self.histories.append(list(history or []))
        return GroundedAnswer(
            status="answered",
            answer="Después del tercer reintento se escala a Treasury Integrations.",
            citation_ids=["E1"],
        )


def test_chat_route_skips_knowledge_search() -> None:
    chat = FakeChat()
    knowledge = FakeKnowledge()
    agent = ConversationalAgent(chat=chat, router=FakeRouter(), knowledge=knowledge)  # type: ignore[arg-type]

    response = agent.invoke(QueryRequest(question="Hola, ¿qué haces?", thread_id="thread-1"))

    assert response.route == "chat"
    assert response.answer == "Soy RAG Ops Guard."
    assert response.citations == []
    assert knowledge.queries == []
    assert chat.chat_calls == 1
    assert chat.answer_calls == 0


def test_knowledge_route_returns_validated_citation() -> None:
    chat = FakeChat()
    knowledge = FakeKnowledge()
    agent = ConversationalAgent(chat=chat, router=FakeRouter(), knowledge=knowledge)  # type: ignore[arg-type]

    response = agent.invoke(
        QueryRequest(question="¿Qué pasa después del tercer reintento?", thread_id="thread-2")
    )

    assert response.route == "knowledge"
    assert response.citations[0].logical_id == "payment-retry-policy"
    assert chat.answer_calls == 1
    assert len(knowledge.queries) == 1


def test_same_thread_keeps_previous_user_subject_for_followup_search() -> None:
    chat = FakeChat()
    knowledge = FakeKnowledge()
    agent = ConversationalAgent(chat=chat, router=FakeRouter(), knowledge=knowledge)  # type: ignore[arg-type]

    agent.invoke(QueryRequest(question="Contame sobre los reintentos de Calypso", thread_id="thread-3"))
    agent.invoke(QueryRequest(question="¿Y qué pasa después del tercero?", thread_id="thread-3"))

    assert len(knowledge.queries) == 2
    assert "Calypso" in knowledge.queries[1]
    assert "después del tercero" in knowledge.queries[1]
    assert chat.answer_calls == 2
