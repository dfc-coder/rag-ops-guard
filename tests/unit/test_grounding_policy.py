from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from rag_ops_guard.agent.grounding import (
    ConversationState,
    EvidenceWindow,
    GroundedSource,
    TurnPlan,
    TurnPolicy,
    TurnPolicyEngine,
)
from rag_ops_guard.domain.models import QueryContext


def retry_source() -> GroundedSource:
    return GroundedSource(
        title="Payments API v2",
        version="2.0",
        system="payments",
        environment="production",
        section="Retry Contract",
        text=(
            "Transient Calypso timeouts may be retried by the automated client up to three times. "
            "After the third automated retry fails, escalate to Treasury Integrations."
        ),
    )


def grounded_state(*, turn_index: int = 1) -> ConversationState:
    window = EvidenceWindow(
        query="Cuantos reintentos permite Calypso?",
        sources=(retry_source(),),
        created_turn=1,
    )
    return ConversationState(
        turn_index=turn_index,
        topic="payments: Cuantos reintentos permite Calypso?",
        system="payments",
        environment="production",
        last_intent="retrieve",
        last_grounded_query="Cuantos reintentos permite Calypso?",
        evidence=window,
        grounded=window.active(turn_index),
    )


def test_initial_internal_fact_is_retrieved_deterministically() -> None:
    plan = TurnPolicyEngine().plan(
        "Cuantos reintentos permite Calypso?",
        ConversationState(),
        QueryContext(),
    )

    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.retrieval_query == "Cuantos reintentos permite Calypso?"


def test_unknown_named_operational_target_is_not_missed() -> None:
    plan = TurnPolicyEngine().plan(
        "Cuantos retries permite Xarlatan?",
        ConversationState(),
        QueryContext(),
    )

    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.retrieval_query == "Cuantos retries permite Xarlatan?"


def test_generic_retry_code_request_stays_direct() -> None:
    plan = TurnPolicyEngine().plan(
        "Implement a Retry class in Python with exponential backoff.",
        ConversationState(),
        QueryContext(),
    )

    assert plan.policy == TurnPolicy.DIRECT


def test_code_request_that_depends_on_internal_calypso_fact_retrieves() -> None:
    plan = TurnPolicyEngine().plan(
        "Escribe codigo Python que respete los retries de Calypso.",
        ConversationState(),
        QueryContext(),
    )

    assert plan.policy == TurnPolicy.RETRIEVE


def test_contextual_new_fact_retrieves_with_prior_grounded_query() -> None:
    plan = TurnPolicyEngine().plan(
        "Y despues del tercero?",
        grounded_state(),
        QueryContext(),
    )

    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.preserve_evidence is True
    assert plan.retrieval_query == (
        "Cuantos reintentos permite Calypso. Y despues del tercero?"
    )
    assert "Follow-up" not in (plan.retrieval_query or "")


def test_explicit_new_internal_target_does_not_inherit_previous_query() -> None:
    plan = TurnPolicyEngine().plan(
        "Que incidentes tiene SendGrid?",
        grounded_state(),
        QueryContext(),
    )

    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.preserve_evidence is False
    assert plan.retrieval_query == "Que incidentes tiene SendGrid?"


def test_transform_reuses_active_evidence_without_new_retrieval() -> None:
    plan = TurnPolicyEngine().plan("Resumilo en una línea", grounded_state(), QueryContext())

    assert plan.policy == TurnPolicy.REUSE_EVIDENCE
    assert plan.preserve_evidence is True
    assert plan.retrieval_query is None
    assert "Payments API v2" in (plan.evidence_context or "")
    assert "untrusted data" in (plan.evidence_context or "")


def test_social_message_after_grounding_stays_direct_and_keeps_topic() -> None:
    plan = TurnPolicyEngine().plan("Gracias", grounded_state(), QueryContext())

    assert plan.policy == TurnPolicy.DIRECT
    assert plan.preserve_evidence is True


def test_general_definition_after_grounding_stays_direct_and_changes_topic() -> None:
    plan = TurnPolicyEngine().plan(
        "Que es exponential backoff?",
        grounded_state(),
        QueryContext(),
    )

    assert plan.policy == TurnPolicy.DIRECT
    assert plan.preserve_evidence is False


def test_explicit_catalog_request_uses_list_policy() -> None:
    plan = TurnPolicyEngine().plan(
        "Que documentacion hay?",
        ConversationState(),
        QueryContext(),
    )

    assert plan.policy == TurnPolicy.LIST_KNOWLEDGE


def test_expired_evidence_is_not_reused_for_elliptical_message() -> None:
    state = grounded_state(turn_index=6)
    assert state.active_evidence() is None

    plan = TurnPolicyEngine().plan("Y despues?", state, QueryContext())

    assert plan.policy == TurnPolicy.DIRECT


def test_successful_tool_payload_commits_explicit_grounding_state() -> None:
    source = retry_source()
    payload = {
        "supported": True,
        "query": "Cuantos reintentos permite Calypso?",
        "sources": [
            {
                "title": source.title,
                "version": source.version,
                "system": source.system,
                "environment": source.environment,
                "section": source.section,
                "text": source.text,
            }
        ],
    }
    messages = [
        HumanMessage(content="Cuantos reintentos permite Calypso?"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_knowledge",
                    "args": {"query": payload["query"]},
                    "id": "search-1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content=json.dumps(payload),
            tool_call_id="search-1",
            name="search_knowledge",
        ),
        AIMessage(content="Calypso permite tres reintentos."),
    ]

    updated = ConversationState().after_success(
        plan=TurnPlan(
            TurnPolicy.RETRIEVE,
            "internal fact",
            retrieval_query=payload["query"],
        ),
        messages=messages,
        context=QueryContext(),
    )

    assert updated.turn_index == 1
    assert updated.grounded is True
    assert updated.system == "payments"
    assert updated.last_grounded_query == payload["query"]
    assert updated.evidence is not None
    assert updated.evidence.sources[0].title == "Payments API v2"


def test_social_success_keeps_window_but_general_topic_shift_clears_it() -> None:
    engine = TurnPolicyEngine()
    state = grounded_state(turn_index=4)

    social_plan = engine.plan("Gracias", state, QueryContext())
    after_social = state.after_success(
        plan=social_plan,
        messages=[HumanMessage(content="Gracias"), AIMessage(content="De nada")],
        context=QueryContext(),
    )
    assert after_social.turn_index == 5
    assert after_social.evidence is not None
    assert after_social.grounded is True

    general_plan = engine.plan("Que es exponential backoff?", after_social, QueryContext())
    after_general = after_social.after_success(
        plan=general_plan,
        messages=[
            HumanMessage(content="Que es exponential backoff?"),
            AIMessage(content="Es una estrategia de espera incremental."),
        ],
        context=QueryContext(),
    )
    assert after_general.turn_index == 6
    assert after_general.evidence is None
    assert after_general.grounded is False
    assert after_general.topic is None
