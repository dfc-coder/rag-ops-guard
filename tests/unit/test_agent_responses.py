from rag_ops_guard.agent.responses import capabilities_response, out_of_scope_response


def test_capabilities_are_application_defined_not_model_invented() -> None:
    answer = capabilities_response("¿Qué puedes hacer?")

    assert "knowledge base" in answer
    assert "citas" in answer
    assert "pagues" not in answer.casefold()
    assert "archivos locales" not in answer.casefold()


def test_out_of_scope_response_keeps_operational_scope() -> None:
    answer = out_of_scope_response("¿Cuál es la capital de Francia?")

    assert "operaciones de integración" in answer
    assert "Francia" not in answer
