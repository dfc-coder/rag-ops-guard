from __future__ import annotations

import json
from dataclasses import replace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from rag_ops_guard.agent.grounding import (
    ConversationState,
    EvidenceWindow,
    GroundedSource,
    GroundingController,
    TurnPolicy,
    TurnPolicyEngine,
)
from rag_ops_guard.agent.semantic_gate import (
    GateDecision,
    GroundingAction,
    SemanticGateContext,
)
from rag_ops_guard.domain.models import QueryContext


PROD = QueryContext(environment="production")
ROOT = "Cuantos reintentos permite Calypso?"


def _source(system: str = "calypso") -> GroundedSource:
    return GroundedSource(
        title="Payment Retry Policy",
        version="2.0",
        system=system,
        environment="production",
        section="Escalation",
        text="Three retries are allowed. After the third failure, escalate to Treasury.",
    )


def _state() -> ConversationState:
    return ConversationState(
        turn_index=1,
        topic="calypso: retry policy",
        system="calypso",
        environment="production",
        last_intent="retrieve",
        last_grounded_query=ROOT,
        evidence=EvidenceWindow(
            query=ROOT,
            sources=(_source(),),
            created_turn=1,
            context_environment="production",
        ),
        grounded=True,
        last_retrieval_supported=True,
        last_user_message=ROOT,
        last_assistant_message="Calypso permite tres reintentos automaticos.",
    )


def _decision(action: GroundingAction) -> GateDecision:
    return GateDecision(action=action, score=0.9, margin=0.4, scores={})


def test_retrieve_uses_stable_context_plus_literal_ranking_query() -> None:
    plan = GroundingController().plan(
        "Y despues del tercero?",
        _decision(GroundingAction.RETRIEVE),
        _state(),
        PROD,
    )

    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.retrieval_query == "Cuantos reintentos permite Calypso. Y despues del tercero?"
    assert plan.ranking_query == "Y despues del tercero?"
    assert plan.evidence_context is None
    assert plan.preserve_evidence is False
    assert plan.preserve_topic is True


def test_first_grounded_request_uses_literal_message_only() -> None:
    plan = GroundingController().plan(
        "What is our current production retry policy?",
        _decision(GroundingAction.RETRIEVE),
        ConversationState(),
        PROD,
    )

    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.retrieval_query == "What is our current production retry policy?"
    assert plan.ranking_query == "What is our current production retry policy?"
    assert plan.preserve_topic is False


def test_direct_turn_uses_history_and_preserves_eligible_grounding_state() -> None:
    plan = GroundingController().plan(
        "Resumilo en una linea.",
        _decision(GroundingAction.DIRECT),
        _state(),
        PROD,
    )

    assert plan.policy == TurnPolicy.DIRECT
    assert plan.retrieval_query is None
    assert plan.evidence_context is None
    assert plan.preserve_evidence is True
    assert plan.preserve_topic is True


def test_catalog_is_a_deterministic_control_path() -> None:
    plan = GroundingController().plan(
        "catalog",
        _decision(GroundingAction.CATALOG),
        _state(),
        PROD,
    )

    assert plan.policy == TurnPolicy.LIST_KNOWLEDGE
    assert plan.preserve_evidence is True
    assert plan.preserve_topic is True


def test_uncertain_gate_result_fails_closed_to_retrieval() -> None:
    plan = GroundingController().plan(
        "ambiguous request",
        _decision(GroundingAction.UNCERTAIN),
        _state(),
        PROD,
    )

    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.ranking_query == "ambiguous request"


def test_environment_change_prevents_evidence_preservation() -> None:
    staging = QueryContext(environment="staging")
    plan = GroundingController().plan(
        "summarize the previous answer",
        _decision(GroundingAction.DIRECT),
        _state(),
        staging,
    )

    assert plan.policy == TurnPolicy.DIRECT
    assert plan.preserve_evidence is False
    assert plan.preserve_topic is True


class FailingGate:
    def decide(
        self,
        message: str,
        context: SemanticGateContext,
    ) -> GateDecision:
        del message, context
        raise RuntimeError("gate unavailable")


def test_semantic_gate_failure_fails_closed() -> None:
    engine = TurnPolicyEngine(gate=FailingGate())

    plan = engine.plan("unknown request", _state(), PROD)

    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.ranking_query == "unknown request"
    assert ROOT.rstrip("?") in (plan.retrieval_query or "")


def test_successful_topic_switch_uses_literal_ranking_query_as_new_root() -> None:
    state = _state()
    plan = GroundingController().plan(
        "Cuantos retries permite Xarlatan?",
        _decision(GroundingAction.RETRIEVE),
        state,
        PROD,
    )
    payload = {
        "supported": True,
        "query": plan.retrieval_query,
        "ranking_query": plan.ranking_query,
        "sources": [
            {
                "title": "Xarlatan Retry Policy",
                "version": "1.0",
                "system": "xarlatan",
                "environment": "production",
                "section": "Retries",
                "text": "Xarlatan retries twice.",
            }
        ],
    }
    messages = [
        HumanMessage(content="Cuantos retries permite Xarlatan?"),
        ToolMessage(
            content=json.dumps(payload),
            name="search_knowledge",
            tool_call_id="search",
        ),
        AIMessage(content="Xarlatan retries twice."),
    ]

    next_state = state.after_success(plan=plan, messages=messages, context=PROD)

    assert next_state.system == "xarlatan"
    assert next_state.last_grounded_query == "Cuantos retries permite Xarlatan?"


def test_direct_turn_does_not_refresh_expired_evidence() -> None:
    state = replace(_state(), turn_index=10, grounded=False)
    plan = GroundingController().plan(
        "summarize previous answer",
        _decision(GroundingAction.DIRECT),
        state,
        PROD,
    )

    assert plan.preserve_evidence is False
    assert plan.preserve_topic is True
