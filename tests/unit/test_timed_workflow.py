from rag_ops_guard.domain.models import GroundedAnswer, QueryAnalysis, QueryContext, QueryRequest
from rag_ops_guard.graph.timed_workflow import TimedRagWorkflow
from rag_ops_guard.retrieval.resolver import EvidenceResolver
from tests.fixtures.builders import evidence
from tests.fixtures.fakes import FakeChatModel, FakeEmbeddingProvider, FakeVectorStore


def test_timed_workflow_reports_stage_latencies() -> None:
    item = evidence()
    chat = FakeChatModel(
        analysis=QueryAnalysis(
            normalized_question="How many retries are allowed?",
            systems=["payments"],
            environment="production",
            api_version=None,
            requires_clarification=False,
            clarification_question=None,
            safety_category="normal",
            fallback_message="The available documentation is insufficient.",
        ),
        answer=GroundedAnswer(
            status="answered",
            answer="The current policy allows three retries.",
            citation_ids=[item.chunk.id],
        ),
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

    expected = {"analysis", "embedding", "retrieval", "resolver", "generation", "total"}
    assert expected.issubset(response.timings_ms)
    assert all(response.timings_ms[key] >= 0 for key in expected)
