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


PROD = QueryContext(environment="production")
STAGING = QueryContext(environment="staging")


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


def grounded_state(*, turn_index: int = 1, context: QueryContext = PROD) -> ConversationState:
    window = EvidenceWindow(
        query="Cuantos reintentos permite Calypso?",
        sources=(retry_source(),),
        created_turn=1,
        context_system=context.system,
        context_environment=context.environment,
        context_api_version=context.api_version,
    )
    return ConversationState(
        turn_index=turn_index,
        topic="payments: Cuantos reintentos permite Calypso?",
        system="payments",
        environment=context.environment,
        last_intent="retrieve",
        last_grounded_query="Cuantos reintentos permite Calypso?",
        evidence=window,
        grounded=window.active(turn_index),
        last_retrieval_supported=True,
    )


def tool_messages(query: str, *, supported: bool = True) -> list:
    payload: dict[str, object] = {"supported": supported, "query": query}
    if supported:
        source = retry_source()
        payload["sources"] = [
            {
                "title": source.title,
                "version": source.version,
                "system": source.system,
                "environment": source.environment,
                "section": source.section,
                "text": source.text,
            }
        ]
    return [
        HumanMessage(content="turn"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_knowledge",
                    "args": {"query": query},
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
        AIMessage(content="answer"),
    ]


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


def test_code_request_that_depends_on_internal_fact_retrieves() -> None:
    plan = TurnPolicyEngine().plan(
        "Escribe codigo Python que respete los retries de Calypso.",
        ConversationState(),
        QueryContext(),
    )
    assert plan.policy == TurnPolicy.RETRIEVE


def test_contextual_new_fact_retrieves_with_natural_stable_query() -> None:
    plan = TurnPolicyEngine().plan("Y despues del tercero?", grounded_state(), PROD)
    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.preserve_evidence is True
    assert plan.preserve_topic is True
    assert plan.retrieval_query == "Cuantos reintentos permite Calypso. Y despues del tercero?"
    assert "Follow-up" not in (plan.retrieval_query or "")


def test_second_followup_does_not_accumulate_previous_followup_text() -> None:
    state = grounded_state()
    first = TurnPolicyEngine().plan("Y despues del tercero?", state, PROD)
    updated = state.after_success(
        plan=first,
        messages=tool_messages(first.retrieval_query or ""),
        context=PROD,
    )
    assert updated.last_grounded_query == "Cuantos reintentos permite Calypso?"

    second = TurnPolicyEngine().plan("Y quien interviene?", updated, PROD)
    assert second.policy == TurnPolicy.RETRIEVE
    assert second.retrieval_query == "Cuantos reintentos permite Calypso. Y quien interviene?"
    assert "despues del tercero" not in (second.retrieval_query or "").casefold()


def test_known_explicit_new_target_does_not_inherit_previous_query() -> None:
    plan = TurnPolicyEngine().plan("Que incidentes tiene SendGrid?", grounded_state(), PROD)
    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.preserve_evidence is False
    assert plan.preserve_topic is False
    assert plan.retrieval_query == "Que incidentes tiene SendGrid?"


def test_unknown_explicit_new_target_does_not_inherit_previous_query() -> None:
    plan = TurnPolicyEngine().plan("Cuantos retries permite Xarlatan?", grounded_state(), PROD)
    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.preserve_topic is False
    assert plan.retrieval_query == "Cuantos retries permite Xarlatan?"
    assert "Calypso" not in (plan.retrieval_query or "")


def test_transform_reuses_active_compatible_evidence_without_retrieval() -> None:
    plan = TurnPolicyEngine().plan("Resumilo en una línea", grounded_state(), PROD)
    assert plan.policy == TurnPolicy.REUSE_EVIDENCE
    assert plan.preserve_evidence is True
    assert plan.preserve_topic is True
    assert plan.retrieval_query is None
    assert "Payments API v2" in (plan.evidence_context or "")


def test_expired_evidence_retrieves_instead_of_answering_directly() -> None:
    state = grounded_state(turn_index=6)
    assert state.active_evidence(PROD) is None
    plan = TurnPolicyEngine().plan("Y despues?", state, PROD)
    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.preserve_evidence is False
    assert plan.preserve_topic is True
    assert "Calypso" in (plan.retrieval_query or "")


def test_expired_evidence_transform_refreshes_before_answering() -> None:
    state = grounded_state(turn_index=6)
    plan = TurnPolicyEngine().plan("Resumilo", state, PROD)
    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.preserve_topic is True


def test_environment_change_invalidates_reuse_and_forces_refresh() -> None:
    state = grounded_state(context=PROD)
    assert state.active_evidence(STAGING) is None
    plan = TurnPolicyEngine().plan("Resumilo", state, STAGING)
    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.preserve_evidence is False


def test_api_version_change_invalidates_reuse() -> None:
    v1 = QueryContext(environment="production", api_version="v1")
    v2 = QueryContext(environment="production", api_version="v2")
    state = grounded_state(context=v1)
    assert state.active_evidence(v2) is None
    plan = TurnPolicyEngine().plan("Resumilo", state, v2)
    assert plan.policy == TurnPolicy.RETRIEVE


def test_same_context_keeps_reuse_available() -> None:
    context = QueryContext(system="payments", environment="production", api_version="v2")
    state = grounded_state(context=context)
    assert state.active_evidence(context) is not None
    plan = TurnPolicyEngine().plan("Explicalo mas corto", state, context)
    assert plan.policy == TurnPolicy.REUSE_EVIDENCE


def test_unsupported_contextual_retrieval_does_not_replace_previous_evidence() -> None:
    state = grounded_state()
    plan = TurnPolicyEngine().plan("Y despues del tercero?", state, PROD)
    updated = state.after_success(
        plan=plan,
        messages=tool_messages(plan.retrieval_query or "", supported=False),
        context=PROD,
    )
    assert updated.last_retrieval_supported is False
    assert updated.last_grounded_query == state.last_grounded_query
    assert updated.evidence == state.evidence

    # Ambiguous transformations must not silently summarize stale evidence after an unsupported
    # new-fact retrieval.
    next_plan = TurnPolicyEngine().plan("Resumilo", updated, PROD)
    assert next_plan.policy == TurnPolicy.DIRECT


def test_successful_contextual_retrieval_keeps_root_topic_but_refreshes_evidence() -> None:
    state = grounded_state()
    plan = TurnPolicyEngine().plan("Y despues del tercero?", state, PROD)
    updated = state.after_success(
        plan=plan,
        messages=tool_messages(plan.retrieval_query or ""),
        context=PROD,
    )
    assert updated.last_retrieval_supported is True
    assert updated.last_grounded_query == "Cuantos reintentos permite Calypso?"
    assert updated.evidence is not None
    assert updated.evidence.query == plan.retrieval_query


def test_social_message_stays_direct_and_preserves_topic() -> None:
    state = grounded_state()
    plan = TurnPolicyEngine().plan("Gracias", state, PROD)
    assert plan.policy == TurnPolicy.DIRECT
    assert plan.preserve_evidence is True
    assert plan.preserve_topic is True

    updated = state.after_success(
        plan=plan,
        messages=[HumanMessage(content="Gracias"), AIMessage(content="De nada")],
        context=PROD,
    )
    assert updated.last_grounded_query == state.last_grounded_query
    assert updated.evidence is not None


def test_general_definition_stays_direct_and_clears_old_topic_after_success() -> None:
    state = grounded_state()
    plan = TurnPolicyEngine().plan("Que es exponential backoff?", state, PROD)
    assert plan.policy == TurnPolicy.DIRECT
    assert plan.preserve_topic is False

    updated = state.after_success(
        plan=plan,
        messages=[
            HumanMessage(content="Que es exponential backoff?"),
            AIMessage(content="Es una estrategia de espera incremental."),
        ],
        context=PROD,
    )
    assert updated.topic is None
    assert updated.last_grounded_query is None
    assert updated.evidence is None


def test_explicit_catalog_request_uses_list_policy() -> None:
    plan = TurnPolicyEngine().plan("Que documentacion hay?", ConversationState(), QueryContext())
    assert plan.policy == TurnPolicy.LIST_KNOWLEDGE


def test_successful_initial_tool_payload_commits_context_bound_grounding_state() -> None:
    query = "Cuantos reintentos permite Calypso?"
    updated = ConversationState().after_success(
        plan=TurnPlan(TurnPolicy.RETRIEVE, "internal fact", retrieval_query=query),
        messages=tool_messages(query),
        context=PROD,
    )
    assert updated.turn_index == 1
    assert updated.grounded is True
    assert updated.system == "payments"
    assert updated.last_grounded_query == query
    assert updated.last_retrieval_supported is True
    assert updated.evidence is not None
    assert updated.evidence.context_environment == "production"
    assert updated.evidence.sources[0].title == "Payments API v2"
