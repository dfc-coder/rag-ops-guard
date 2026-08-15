from __future__ import annotations

from dataclasses import replace

from rag_ops_guard.agent.grounding import (
    ConversationState,
    EvidenceWindow,
    GroundedSource,
    GroundingController,
    TurnPolicy,
    TurnPolicyEngine,
)
from rag_ops_guard.agent.semantic_router import (
    ContextRelation,
    SemanticConversationContext,
    TurnDecision,
    TurnOperation,
)
from rag_ops_guard.domain.models import QueryContext


PROD = QueryContext(environment="production")


def _source() -> GroundedSource:
    return GroundedSource(
        title="Payment Retry Policy",
        version="2.0",
        system="payments",
        environment="production",
        section="Escalation",
        text="Three retries are allowed. After the third failure, escalate to Treasury.",
    )


def _state() -> ConversationState:
    return ConversationState(
        turn_index=1,
        topic="payments: retry policy",
        system="payments",
        environment="production",
        last_intent="retrieve",
        last_grounded_query="How many retries are allowed?",
        evidence=EvidenceWindow(
            query="How many retries are allowed?",
            sources=(_source(),),
            created_turn=1,
            context_environment="production",
        ),
        grounded=True,
        last_retrieval_supported=True,
        last_user_message="How many retries are allowed?",
        last_assistant_message="Three retries are allowed.",
    )


def _decision(
    *,
    grounding: bool,
    relation: ContextRelation,
    operation: TurnOperation = TurnOperation.ANSWER,
    query: str | None = None,
) -> TurnDecision:
    return TurnDecision(
        requires_grounding=grounding,
        relation_to_context=relation,
        operation=operation,
        standalone_query=query,
    )


def test_same_context_new_fact_retrieves_semantic_standalone_query() -> None:
    plan = GroundingController().plan(
        "What happens next?",
        _decision(
            grounding=True,
            relation=ContextRelation.SAME,
            query="What happens after the retry limit is reached?",
        ),
        _state(),
        PROD,
    )

    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.retrieval_query == "What happens after the retry limit is reached?"
    assert plan.ranking_query == "What happens next?"
    assert plan.preserve_topic is True
    assert plan.preserve_evidence is True
    assert plan.evidence_context is not None


def test_grounding_requirement_dominates_transform_reuse() -> None:
    plan = GroundingController().plan(
        "Explain a new internal fact while transforming it",
        _decision(
            grounding=True,
            relation=ContextRelation.SAME,
            operation=TurnOperation.TRANSFORM,
            query="Self-contained internal fact query",
        ),
        _state(),
        PROD,
    )

    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.retrieval_query == "Self-contained internal fact query"
    assert plan.ranking_query == "Explain a new internal fact while transforming it"


def test_new_grounded_subject_does_not_inherit_previous_topic() -> None:
    plan = GroundingController().plan(
        "Question about another system",
        _decision(
            grounding=True,
            relation=ContextRelation.NEW,
            query="Question about another system",
        ),
        _state(),
        PROD,
    )

    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.retrieval_query == "Question about another system"
    assert plan.preserve_topic is False
    assert plan.preserve_evidence is False
    assert plan.evidence_context is None


def test_transform_same_context_reuses_active_evidence() -> None:
    plan = GroundingController().plan(
        "Transform the previous answer",
        _decision(
            grounding=False,
            relation=ContextRelation.SAME,
            operation=TurnOperation.TRANSFORM,
        ),
        _state(),
        PROD,
    )

    assert plan.policy == TurnPolicy.REUSE_EVIDENCE
    assert plan.retrieval_query is None
    assert plan.preserve_topic is True
    assert plan.preserve_evidence is True
    assert plan.evidence_context is not None


def test_transform_refreshes_root_when_evidence_expired() -> None:
    state = replace(_state(), turn_index=10, grounded=False)
    plan = GroundingController().plan(
        "Transform the previous answer",
        _decision(
            grounding=False,
            relation=ContextRelation.SAME,
            operation=TurnOperation.TRANSFORM,
        ),
        state,
        PROD,
    )

    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.retrieval_query == state.last_grounded_query
    assert plan.ranking_query == state.last_grounded_query
    assert plan.preserve_topic is True


def test_transform_non_grounded_context_stays_direct() -> None:
    plan = GroundingController().plan(
        "Transform the previous answer",
        _decision(
            grounding=False,
            relation=ContextRelation.SAME,
            operation=TurnOperation.TRANSFORM,
        ),
        ConversationState(),
        PROD,
    )

    assert plan.policy == TurnPolicy.DIRECT


def test_general_independent_turn_is_direct_and_does_not_keep_old_grounding() -> None:
    plan = GroundingController().plan(
        "General independent request",
        _decision(grounding=False, relation=ContextRelation.NONE),
        _state(),
        PROD,
    )

    assert plan.policy == TurnPolicy.DIRECT
    assert plan.preserve_topic is False
    assert plan.preserve_evidence is False


def test_same_context_non_grounded_turn_preserves_active_topic() -> None:
    plan = GroundingController().plan(
        "Acknowledgement related to current discussion",
        _decision(grounding=False, relation=ContextRelation.SAME),
        _state(),
        PROD,
    )

    assert plan.policy == TurnPolicy.DIRECT
    assert plan.preserve_topic is True
    assert plan.preserve_evidence is True


def test_catalog_is_deterministic_and_preserves_existing_context() -> None:
    plan = GroundingController().plan(
        "catalog request",
        _decision(
            grounding=True,
            relation=ContextRelation.NONE,
            operation=TurnOperation.CATALOG,
            query="catalog request",
        ),
        _state(),
        PROD,
    )

    assert plan.policy == TurnPolicy.LIST_KNOWLEDGE
    assert plan.preserve_topic is True
    assert plan.preserve_evidence is True


def test_context_filter_change_invalidates_evidence_reuse() -> None:
    staging = QueryContext(environment="staging")
    plan = GroundingController().plan(
        "Transform previous grounded answer",
        _decision(
            grounding=False,
            relation=ContextRelation.SAME,
            operation=TurnOperation.TRANSFORM,
        ),
        _state(),
        staging,
    )

    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.retrieval_query == _state().last_grounded_query
    assert plan.evidence_context is None


class FailingResolver:
    def resolve(
        self,
        message: str,
        context: SemanticConversationContext,
    ) -> TurnDecision:
        del message, context
        raise RuntimeError("router unavailable")


def test_semantic_router_failure_fails_closed_with_grounded_context() -> None:
    engine = TurnPolicyEngine(resolver=FailingResolver())  # type: ignore[arg-type]

    plan = engine.plan("ambiguous request", _state(), PROD)

    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.preserve_topic is True
    assert "How many retries are allowed" in (plan.retrieval_query or "")
    assert "ambiguous request" in (plan.retrieval_query or "")


def test_semantic_router_failure_fails_closed_without_context() -> None:
    engine = TurnPolicyEngine(resolver=FailingResolver())  # type: ignore[arg-type]

    plan = engine.plan("unknown request", ConversationState(), PROD)

    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.retrieval_query == "unknown request"
    assert plan.preserve_topic is False
