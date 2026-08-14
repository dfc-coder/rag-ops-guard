from rag_ops_guard.domain.models import (
    GroundedAnswer,
    QueryAnalysis,
    QueryContext,
    QueryRequest,
    QueryStatus,
)
from rag_ops_guard.graph.timed_workflow import TimedRagWorkflow
from rag_ops_guard.retrieval.resolver import EvidenceResolver
from tests.fixtures.builders import evidence
from tests.fixtures.fakes import FakeChatModel, FakeEmbeddingProvider, FakeVectorStore


def _chat(answer: GroundedAnswer) -> FakeChatModel:
    return FakeChatModel(
        analysis=QueryAnalysis(
            normalized_question="unused",
            systems=[],
            environment=None,
            api_version=None,
            requires_clarification=False,
            clarification_question=None,
            safety_category="normal",
        ),
        answer=answer,
    )


def test_timed_workflow_uses_one_llm_call_and_reports_stage_latencies() -> None:
    item = evidence()
    chat = _chat(
        GroundedAnswer(
            status="answered",
            answer="The current policy allows three retries.",
            citation_ids=["E1"],
        )
    )
    workflow = TimedRagWorkflow(
        chat=chat,
        embeddings=FakeEmbeddingProvider(),
        vectors=FakeVectorStore(evidence=[item]),
        resolver=EvidenceResolver(),
    )

    response = workflow.invoke(
        QueryRequest(
            question="How many retries are allowed?",
            context=QueryContext(system="payments", environment="production"),
        )
    )

    expected = {"embedding", "retrieval", "resolver", "generation", "total"}
    assert response.status == QueryStatus.ANSWERED
    assert expected.issubset(response.timings_ms)
    assert "analysis" not in response.timings_ms
    assert all(response.timings_ms[key] >= 0 for key in expected)
    assert response.citations[0].chunk_id == item.chunk.id
    assert chat.analysis_calls == 0
    assert chat.generation_calls == 1


def test_timed_workflow_maps_model_clarification_without_second_call() -> None:
    chat = _chat(
        GroundedAnswer(
            status="clarification_required",
            answer="Which environment do you mean?",
            citation_ids=[],
        )
    )
    workflow = TimedRagWorkflow(
        chat=chat,
        embeddings=FakeEmbeddingProvider(),
        vectors=FakeVectorStore(),
        resolver=EvidenceResolver(),
    )

    response = workflow.invoke(QueryRequest(question="What timeout applies?"))

    assert response.status == QueryStatus.CLARIFICATION_REQUIRED
    assert response.clarification_question == "Which environment do you mean?"
    assert chat.analysis_calls == 0
    assert chat.generation_calls == 1
