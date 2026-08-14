from rag_ops_guard.adapters.llm.llamacpp_chat import _trusted_followup_fallback


def test_trusted_followup_fallback_uses_only_grounded_context() -> None:
    query = _trusted_followup_fallback(
        current_question="Y despues del tercero?",
        previous_query="Cuantos reintentos permite Calypso?",
        source_titles=["Payment Retry Policy", "Payment Retry Policy"],
    )

    assert query == (
        "Cuantos reintentos permite Calypso?\n"
        "Y despues del tercero?\n"
        "Relevant sources: Payment Retry Policy"
    )
