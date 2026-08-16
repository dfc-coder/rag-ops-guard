from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ENTRYPOINTS = (
    ROOT / "src/rag_ops_guard/app.py",
    ROOT / "src/rag_ops_guard/handlers/query.py",
    ROOT / "scripts/chainlit_react_ui.py",
    ROOT / "scripts/gradio_react_ui.py",
)
LEGACY_MODULES = {
    "rag_ops_guard.agent.react_agent",
    "rag_ops_guard.graph.conversational_agent",
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_runtime_entrypoints_do_not_import_legacy_orchestrators() -> None:
    offenders: list[str] = []
    for path in RUNTIME_ENTRYPOINTS:
        overlap = _imports(path).intersection(LEGACY_MODULES)
        if overlap:
            offenders.append(f"{path.relative_to(ROOT)} -> {sorted(overlap)}")
    assert not offenders, f"runtime bifurcation reintroduced: {offenders}"


def test_all_runtime_clients_resolve_the_canonical_agent_from_app() -> None:
    chainlit = (ROOT / "scripts/chainlit_react_ui.py").read_text(encoding="utf-8")
    gradio = (ROOT / "scripts/gradio_react_ui.py").read_text(encoding="utf-8")
    app = (ROOT / "src/rag_ops_guard/app.py").read_text(encoding="utf-8")

    assert "from rag_ops_guard.app import conversation_agent" in chainlit
    assert "AGENT = conversation_agent()" in chainlit
    assert "from rag_ops_guard.app import conversation_agent" in gradio
    assert "AGENT = conversation_agent()" in gradio
    assert "def conversation_agent() -> ConversationAgent:" in app
    assert "return conversation_agent()" in app


def test_api_query_workflow_is_only_a_compatibility_alias_to_canonical_agent() -> None:
    app = (ROOT / "src/rag_ops_guard/app.py").read_text(encoding="utf-8")
    handler = (ROOT / "src/rag_ops_guard/handlers/query.py").read_text(encoding="utf-8")

    assert "def query_workflow() -> ConversationAgent:" in app
    assert "return conversation_agent()" in app
    assert "response = query_workflow().invoke(request)" in handler


def test_canonical_agent_uses_tool_calling_port_without_fabricated_calls() -> None:
    source = (ROOT / "src/rag_ops_guard/agent/conversation.py").read_text(encoding="utf-8")

    assert "model.bind_tools(tools)" in source
    assert '"name": "search_documents"' not in source
    assert "TurnPolicyEngine" not in source
    assert "SemanticGroundingGate" not in source
