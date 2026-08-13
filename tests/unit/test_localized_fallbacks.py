from rag_ops_guard.domain.models import (
    GroundedAnswer,
    QueryAnalysis,
    QueryRequest,
    QueryStatus,
)
from rag_ops_guard.graph.workflow import RagWorkflow
from rag_ops_guard.retrieval.resolver import EvidenceResolver
from tests.fixtures.fakes import FakeChatModel, FakeEmbeddingProvider, FakeVectorStore


def workflow(chat: FakeChatModel) -> RagWorkflow:
    return RagWorkflow(
        chat=chat,
        embeddings=FakeEmbeddingProvider(),
        vectors=FakeVectorStore(),
        resolver=EvidenceResolver(),
    )


def test_insufficient_evidence_preserves_question_language() -> None:
    message = "La documentación admitida no aporta evidencia suficiente para responder con seguridad."
    chat = FakeChatModel(
        analysis=QueryAnalysis(
            normalized_question="¿Cuál es el timeout de SAP en producción?",
            systems=["sap"],
            environment="production",
            api_version=None,
            requires_clarification=False,
            clarification_question=None,
            safety_category="normal",
            safety_blocked_message="La solicitud fue bloqueada por seguridad.",
            insufficient_evidence_message=message,
        ),
        answer=GroundedAnswer(status="answered", answer="unused", citation_ids=[]),
    )

    response = workflow(chat).invoke(
        QueryRequest(question="¿Cuál es el timeout de SAP en producción?")
    )

    assert response.status == QueryStatus.INSUFFICIENT_EVIDENCE
    assert response.answer == message
    assert chat.generation_calls == 0


def test_safety_node_uses_prelocalized_message() -> None:
    chat = FakeChatModel(
        analysis=QueryAnalysis(
            normalized_question="consulta",
            systems=[],
            environment=None,
            api_version=None,
            requires_clarification=False,
            clarification_question=None,
            safety_category="normal",
            safety_blocked_message="La solicitud fue bloqueada por seguridad.",
            insufficient_evidence_message="No hay evidencia suficiente.",
        ),
        answer=GroundedAnswer(status="answered", answer="unused", citation_ids=[]),
    )
    rag = workflow(chat)

    result = rag._safety_blocked(
        {
            "safety_blocked_message": "La solicitud fue bloqueada por seguridad.",
            "graph_path": [],
        }
    )

    assert result["answer"] == "La solicitud fue bloqueada por seguridad."
