from __future__ import annotations

from dataclasses import replace

from rag_ops_guard.agent.grounding import (
    ConversationState,
    EvidenceWindow,
    GroundedSource,
    TurnPolicy,
    TurnPolicyEngine,
)
from rag_ops_guard.domain.models import QueryContext


PROD = QueryContext(environment="production")


def _state() -> ConversationState:
    evidence = EvidenceWindow(
        query="Cuantos reintentos permite Calypso?",
        sources=(
            GroundedSource(
                title="Payments API v2",
                version="2.0",
                system="payments",
                environment="production",
                section="Retry Contract",
                text=(
                    "Calypso may be retried up to three times. After the third failure, "
                    "escalate to Treasury Integrations."
                ),
            ),
        ),
        created_turn=1,
        context_environment="production",
    )
    return ConversationState(
        turn_index=1,
        topic="payments: Cuantos reintentos permite Calypso?",
        system="payments",
        environment="production",
        last_intent="retrieve",
        last_grounded_query="Cuantos reintentos permite Calypso?",
        evidence=evidence,
        grounded=True,
        last_retrieval_supported=True,
    )


def test_contextual_code_reuses_active_grounded_evidence() -> None:
    plan = TurnPolicyEngine().plan(
        "Escribe codigo Python para implementar eso.",
        _state(),
        PROD,
    )
    assert plan.policy == TurnPolicy.REUSE_EVIDENCE
    assert plan.preserve_topic is True
    assert plan.preserve_evidence is True
    assert plan.evidence_context is not None


def test_contextual_code_refreshes_when_evidence_expired() -> None:
    state = replace(_state(), turn_index=10, grounded=False)
    plan = TurnPolicyEngine().plan("Implementalo en Python usando eso.", state, PROD)
    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.ranking_query == "Implementalo en Python usando eso."
    assert "Calypso" in (plan.retrieval_query or "")


def test_causal_followup_is_regrounded() -> None:
    plan = TurnPolicyEngine().plan("Por que?", _state(), PROD)
    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.ranking_query == "Por que?"


def test_how_does_followup_is_regrounded() -> None:
    plan = TurnPolicyEngine().plan("How does that work?", _state(), PROD)
    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.ranking_query == "How does that work?"


def test_current_literal_turn_is_kept_separate_for_reranking() -> None:
    plan = TurnPolicyEngine().plan("Y despues del tercero?", _state(), PROD)
    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.ranking_query == "Y despues del tercero?"
    assert plan.retrieval_query == "Cuantos reintentos permite Calypso. Y despues del tercero?"


def test_unrelated_code_after_grounding_stays_direct() -> None:
    plan = TurnPolicyEngine().plan("Escribe Fibonacci en Python.", _state(), PROD)
    assert plan.policy == TurnPolicy.DIRECT


def test_explicit_internal_code_request_retrieves_new_evidence() -> None:
    plan = TurnPolicyEngine().plan(
        "Escribe codigo que implemente los retries de Calypso.",
        _state(),
        PROD,
    )
    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.retrieval_query == "Escribe codigo que implemente los retries de Calypso."


def test_topic_root_is_not_prefixed_to_unknown_explicit_target() -> None:
    plan = TurnPolicyEngine().plan("Cuantos retries permite Xarlatan?", _state(), PROD)
    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.retrieval_query == "Cuantos retries permite Xarlatan?"
    assert "Calypso" not in (plan.retrieval_query or "")


def test_sla_followup_keeps_current_topic_instead_of_treating_sla_as_target() -> None:
    plan = TurnPolicyEngine().plan("Y su SLA?", _state(), PROD)
    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.retrieval_query == "Cuantos reintentos permite Calypso. Y su SLA?"


def test_unknown_lowercase_target_in_retry_question_retrieves() -> None:
    plan = TurnPolicyEngine().plan(
        "cuantos retries permite xarlatan?",
        ConversationState(),
        QueryContext(),
    )
    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.retrieval_query == "cuantos retries permite xarlatan?"
