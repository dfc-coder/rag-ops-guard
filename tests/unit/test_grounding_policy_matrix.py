from __future__ import annotations

import json
from dataclasses import replace

import pytest
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


CONTEXT = QueryContext(environment="production")
ROOT = "Cuantos reintentos permite Calypso?"


def _source() -> GroundedSource:
    return GroundedSource(
        title="Payment Retry Policy",
        version="2.0",
        system="payments",
        environment="production",
        section="Escalation",
        text=(
            "Calypso allows three automated retries. After the third failure, "
            "escalate to Treasury Integrations."
        ),
    )


def _state() -> ConversationState:
    evidence = EvidenceWindow(
        query=ROOT,
        sources=(_source(),),
        created_turn=1,
        context_environment="production",
    )
    return ConversationState(
        turn_index=1,
        topic=f"payments: {ROOT}",
        system="payments",
        environment="production",
        last_intent="retrieve",
        last_grounded_query=ROOT,
        evidence=evidence,
        grounded=True,
        last_retrieval_supported=True,
    )


@pytest.mark.parametrize(
    "message",
    [
        "Y despues?",
        "Que pasa luego?",
        "Y si falla?",
        "Quien interviene?",
        "Cuando se escala?",
        "Por que?",
        "Como funciona eso?",
        "Hay algun limite manual?",
        "Cual es el procedimiento siguiente?",
        "What happens next?",
        "And after the third?",
        "Who handles it?",
        "When does escalation happen?",
        "Why?",
        "How does that work?",
        "Is there a manual limit?",
    ],
)
def test_grounded_fact_followup_variants_retrieve(message: str) -> None:
    plan = TurnPolicyEngine().plan(message, _state(), CONTEXT)
    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.preserve_topic is True
    assert plan.ranking_query == message
    assert "Calypso" in (plan.retrieval_query or "")


@pytest.mark.parametrize(
    "message",
    [
        "Resumilo.",
        "Explicalo mas corto.",
        "Reformula la respuesta.",
        "Dame un ejemplo.",
        "Translate it.",
        "Ponelo en una tabla.",
    ],
)
def test_pure_transform_variants_reuse_active_evidence(message: str) -> None:
    plan = TurnPolicyEngine().plan(message, _state(), CONTEXT)
    assert plan.policy == TurnPolicy.REUSE_EVIDENCE
    assert plan.retrieval_query is None
    assert plan.evidence_context is not None


@pytest.mark.parametrize(
    "message",
    [
        "Explicame que pasa despues del tercero.",
        "Dame un ejemplo de que pasa si falla el tercero.",
        "Resume quien interviene despues.",
        "Explain what happens after the third retry.",
        "Give me an example of what happens when it fails.",
    ],
)
def test_transform_word_does_not_hide_new_fact_intent(message: str) -> None:
    plan = TurnPolicyEngine().plan(message, _state(), CONTEXT)
    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.ranking_query == message


@pytest.mark.parametrize(
    "message",
    [
        "Hola",
        "Gracias",
        "Perfecto",
        "Que es exponential backoff?",
        "What is exponential backoff?",
        "Escribe Fibonacci en Python.",
        "Write a Caesar cipher function in Perl.",
    ],
)
def test_unrelated_direct_variants_do_not_trigger_rag(message: str) -> None:
    plan = TurnPolicyEngine().plan(message, _state(), CONTEXT)
    assert plan.policy == TurnPolicy.DIRECT


@pytest.mark.parametrize(
    "message",
    [
        "Cuantos retries permite Xarlatan?",
        "cuantos retries permite xarlatan?",
        "Escribe codigo que respete los retries de Xarlatan.",
        "Cuantos retries permite SendGrid?",
    ],
)
def test_explicit_new_targets_do_not_inherit_old_topic(message: str) -> None:
    plan = TurnPolicyEngine().plan(message, _state(), CONTEXT)
    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.preserve_topic is False
    assert "Calypso" not in (plan.retrieval_query or "")


def _success_messages(query: str) -> list:
    source = _source()
    payload = {
        "supported": True,
        "query": query,
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
    return [
        HumanMessage(content="follow-up"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_knowledge",
                    "args": {"query": query},
                    "id": "matrix-search",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content=json.dumps(payload),
            name="search_knowledge",
            tool_call_id="matrix-search",
        ),
        AIMessage(content="grounded answer"),
    ]


def test_repeated_followups_never_accumulate_rewritten_query() -> None:
    state = _state()
    engine = TurnPolicyEngine()
    followups = [
        "Y despues?",
        "Y quien interviene?",
        "Y cuando se escala?",
        "Y si falla otra vez?",
        "Y cual es el procedimiento manual?",
    ]

    for message in followups:
        plan = engine.plan(message, state, CONTEXT)
        assert plan.policy == TurnPolicy.RETRIEVE
        assert plan.retrieval_query is not None
        assert plan.retrieval_query.startswith(ROOT.rstrip("?"))
        state = state.after_success(
            plan=plan,
            messages=_success_messages(plan.retrieval_query),
            context=CONTEXT,
        )
        assert state.last_grounded_query == ROOT


def test_expired_window_keeps_topic_but_requires_fresh_retrieval() -> None:
    state = replace(_state(), turn_index=20, grounded=False)
    assert state.active_evidence(CONTEXT) is None

    plan = TurnPolicyEngine().plan("Y despues?", state, CONTEXT)

    assert plan.policy == TurnPolicy.RETRIEVE
    assert plan.preserve_evidence is False
    assert plan.preserve_topic is True
