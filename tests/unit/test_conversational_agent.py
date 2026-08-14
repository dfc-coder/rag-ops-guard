from __future__ import annotations

from langchain_core.messages import BaseMessage

from rag_ops_guard.agent.router import RouteDecision
from rag_ops_guard.domain.models import GroundedAnswer, QueryRequest, QueryStatus
from rag_ops_guard.graph.conversational_agent import ConversationalAgent
from rag_ops_guard.retrieval.hybrid import KnowledgeSearchResult
from tests.fixtures.builders import evidence


class FakeRouter:
    def __init__(self, route: str, scores: dict[str, float] | None = None) -> None:
        self._route = route
        self._scores = scores or {route: 0.9}
        self.questions: list[str] = []

    def route(self, text: str) -> RouteDecision:
        self.questions.append(text)
        return RouteDecision(
            route=self._route,  # type: ignore[arg-type]
            score=max(self._scores.values()),
            margin=0.0 if self._route == "uncertain" else 0.4,
            scores=self._scores,
        )


class FakeCatalog:
    def __init__(self) -> None:
        self.calls = 0

    def render(self, question: str, context: object) -> str:
        del question, context
        self.calls += 1
        return "Documentación activa disponible:\n- Calypso Integration API"


class FakeKnowledge:
    def __init__(self, relevance_by_query: dict[str, float] | None = None) -> None:
        self.queries: list[str] = []
        self.refreshed = False
        self._relevance_by_query = relevance_by_query or {}

    def search(self, query: str, context: object) -> KnowledgeSearchResult:
        del context
        self.queries.append(query)
        item = evidence(text="After the third retry, escalate to Treasury Integrations.")
        relevance = self._relevance_by_query.get(query, 0.9)
        return KnowledgeSearchResult(
            dense=[item],
            lexical=[item],
            fused=[item],
            admitted=[item],
            relevance=relevance,
        )

    def refresh(self) -> None:
        self.refreshed = True


class FakeChat:
    def __init__(self) -> None:
        self.chat_calls = 0
        self.answer_calls = 0
        self.rewrite_calls = 0
        self.answer_histories: list[list[BaseMessage] | None] = []
        self.rewrite_inputs: list[tuple[str, str, list[str]]] = []

    def analyze_query(self, prompt: str) -> object:
        raise AssertionError(prompt)

    def rewrite_query(
        self,
        current_question: str,
        previous_query: str,
        source_titles: list[str],
    ) -> str:
        self.rewrite_calls += 1
        self.rewrite_inputs.append((current_question, previous_query, source_titles))
        return f"{previous_query} {current_question}"

    def generate_chat(self, messages: list[BaseMessage]) -> str:
        self.chat_calls += 1
        del messages
        return "Hola."

    def generate_answer(
        self,
        prompt: str,
        history: list[BaseMessage] | None = None,
    ) -> GroundedAnswer:
        del prompt
        self.answer_calls += 1
        self.answer_histories.append(history)
        return GroundedAnswer(
            status="answered",
            answer="Después del tercer reintento se escala a Treasury Integrations.",
            citation_ids=["E1"],
        )


def _agent(
    route: str,
    *,
    knowledge: FakeKnowledge | None = None,
    chat: FakeChat | None = None,
    catalog: FakeCatalog | None = None,
    scores: dict[str, float] | None = None,
) -> tuple[ConversationalAgent, FakeKnowledge, FakeChat, FakeCatalog]:
    resolved_knowledge = knowledge or FakeKnowledge()
    resolved_chat = chat or FakeChat()
    resolved_catalog = catalog or FakeCatalog()
    agent = ConversationalAgent(
        chat=resolved_chat,
        router=FakeRouter(route, scores=scores),  # type: ignore[arg-type]
        knowledge=resolved_knowledge,  # type: ignore[arg-type]
        catalog=resolved_catalog,  # type: ignore[arg-type]
        relevance_threshold=0.4,
    )
    return agent, resolved_knowledge, resolved_chat, resolved_catalog


def test_chat_route_skips_knowledge_search() -> None:
    agent, knowledge, chat, catalog = _agent("chat")

    response = agent.invoke(QueryRequest(question="Hola, ¿cómo estás?", thread_id="thread-1"))

    assert response.route == "chat"
    assert response.answer == "Hola."
    assert response.citations == []
    assert knowledge.queries == []
    assert catalog.calls == 0
    assert chat.chat_calls == 1
    assert chat.answer_calls == 0


def test_catalog_route_lists_real_catalog_without_llm() -> None:
    agent, knowledge, chat, catalog = _agent("catalog")

    response = agent.invoke(
        QueryRequest(question="¿Qué documentación tienes disponible?", thread_id="thread-catalog")
    )

    assert response.route == "catalog"
    assert "Calypso Integration API" in (response.answer or "")
    assert catalog.calls == 1
    assert knowledge.queries == []
    assert chat.chat_calls == 0
    assert chat.answer_calls == 0


def test_capabilities_route_is_deterministic_and_skips_llm() -> None:
    agent, knowledge, chat, _ = _agent("capabilities")

    response = agent.invoke(QueryRequest(question="¿Qué puedes hacer?", thread_id="thread-cap"))

    assert response.route == "capabilities"
    assert "knowledge base" in (response.answer or "")
    assert knowledge.queries == []
    assert chat.chat_calls == 0
    assert chat.answer_calls == 0


def test_uncertain_route_uses_real_evidence_to_resolve_to_knowledge() -> None:
    question = "¿Qué pasa con SendGrid?"
    knowledge = FakeKnowledge(relevance_by_query={question: 0.9})
    agent, _, chat, _ = _agent(
        "uncertain",
        knowledge=knowledge,
        scores={"knowledge": 0.63, "out_of_scope": 0.625, "capabilities": 0.2, "chat": 0.1},
    )

    response = agent.invoke(QueryRequest(question=question, thread_id="thread-sendgrid"))

    assert response.route == "knowledge"
    assert response.status == QueryStatus.ANSWERED
    assert response.citations
    assert knowledge.queries == [question]
    assert chat.answer_calls == 1


def test_uncertain_route_with_weak_evidence_falls_back_to_best_control_intent() -> None:
    question = "¿Qué haces?"
    knowledge = FakeKnowledge(relevance_by_query={question: 0.1})
    agent, _, chat, _ = _agent(
        "uncertain",
        knowledge=knowledge,
        scores={"capabilities": 0.62, "knowledge": 0.615, "chat": 0.4, "out_of_scope": 0.2},
    )

    response = agent.invoke(QueryRequest(question=question, thread_id="thread-what-do-you-do"))

    assert response.route == "capabilities"
    assert response.status == QueryStatus.ANSWERED
    assert "knowledge base" in (response.answer or "")
    assert knowledge.queries == [question]
    assert chat.chat_calls == 0
    assert chat.answer_calls == 0


def test_knowledge_search_uses_only_current_question_not_chat_history() -> None:
    knowledge = FakeKnowledge()
    chat = FakeChat()
    catalog = FakeCatalog()
    first = ConversationalAgent(
        chat=chat,
        router=FakeRouter("chat"),  # type: ignore[arg-type]
        knowledge=knowledge,  # type: ignore[arg-type]
        catalog=catalog,  # type: ignore[arg-type]
    )
    first.invoke(QueryRequest(question="Hola, ¿cómo estás?", thread_id="thread-current"))

    first._router = FakeRouter("knowledge")  # type: ignore[attr-defined,assignment]
    question = "¿Cuál es el objetivo de Calypso Payments API?"
    response = first.invoke(QueryRequest(question=question, thread_id="thread-current"))

    assert response.status == QueryStatus.ANSWERED
    assert knowledge.queries == [question]
    assert chat.answer_histories == [None]


def test_low_relevance_followup_rewrites_from_trusted_grounded_focus() -> None:
    first_query = "Contame sobre los reintentos de Calypso"
    followup = "¿Y qué pasa después del tercero?"
    rewritten = f"{first_query} {followup}"
    knowledge = FakeKnowledge(
        relevance_by_query={
            first_query: 0.9,
            followup: 0.1,
            rewritten: 0.9,
        }
    )
    agent, _, chat, _ = _agent("knowledge", knowledge=knowledge)

    agent.invoke(QueryRequest(question=first_query, thread_id="thread-followup"))
    response = agent.invoke(QueryRequest(question=followup, thread_id="thread-followup"))

    assert response.status == QueryStatus.ANSWERED
    assert knowledge.queries == [first_query, followup, rewritten]
    assert response.retrieval_query == rewritten
    assert response.rewritten_query == rewritten
    assert chat.rewrite_calls == 1
    assert chat.rewrite_inputs[0][1] == first_query
    assert chat.answer_histories == [None, None]


def test_low_relevance_without_grounded_focus_abstains_before_generation() -> None:
    question = "¿Qué información existe sobre un sistema desconocido?"
    knowledge = FakeKnowledge(relevance_by_query={question: 0.1})
    agent, _, chat, _ = _agent("knowledge", knowledge=knowledge)

    response = agent.invoke(QueryRequest(question=question, thread_id="thread-low"))

    assert response.status == QueryStatus.INSUFFICIENT_EVIDENCE
    assert response.citations == []
    assert chat.answer_calls == 0
    assert chat.rewrite_calls == 0


def test_out_of_scope_route_does_not_touch_knowledge_or_llm() -> None:
    agent, knowledge, chat, _ = _agent("out_of_scope")

    response = agent.invoke(QueryRequest(question="¿Cuál es la capital de Francia?"))

    assert response.route == "out_of_scope"
    assert response.status == QueryStatus.ANSWERED
    assert knowledge.queries == []
    assert chat.chat_calls == 0
    assert chat.answer_calls == 0
