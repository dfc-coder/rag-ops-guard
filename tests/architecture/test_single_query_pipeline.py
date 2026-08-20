from __future__ import annotations

import ast
import tomllib
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


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name} in {path.relative_to(ROOT)}")


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
    canonical = _function(ROOT / "src/rag_ops_guard/app.py", "conversation_agent")

    assert "from rag_ops_guard.app import conversation_agent" in chainlit
    assert "AGENT = conversation_agent()" in chainlit
    assert "from rag_ops_guard.app import conversation_agent" in gradio
    assert "AGENT = conversation_agent()" in gradio
    assert canonical.args.args[0].arg == "context"
    assert canonical.args.defaults
    assert isinstance(canonical.args.defaults[0], ast.Constant)
    assert canonical.args.defaults[0].value is None


def test_api_uses_authenticated_context_with_canonical_agent() -> None:
    app = (ROOT / "src/rag_ops_guard/app.py").read_text(encoding="utf-8")
    handler = (ROOT / "src/rag_ops_guard/handlers/query.py").read_text(encoding="utf-8")

    assert "def query_workflow(" not in app
    assert "from rag_ops_guard.app import conversation_agent" in handler
    assert "request_context = request_context_from_event(event)" in handler
    assert "response = conversation_agent(request_context).invoke(request)" in handler
    assert "query_workflow" not in handler


def test_canonical_agent_uses_tool_calling_port_without_fabricated_calls() -> None:
    source = (ROOT / "src/rag_ops_guard/agent/conversation.py").read_text(encoding="utf-8")

    assert "model.bind_tools(tools)" in source
    assert '"name": "search_documents"' not in source
    assert "TurnPolicyEngine" not in source
    assert "SemanticGroundingGate" not in source


def test_conversation_agent_is_domain_agnostic_and_reflective() -> None:
    source = (ROOT / "src/rag_ops_guard/agent/conversation.py").read_text(encoding="utf-8")

    assert "KnowledgeSearch" not in source
    assert "KnowledgeCatalog" not in source
    assert "SearchDocumentsTool" not in source
    assert "ListDocumentsTool" not in source
    assert "_probe_relevance" not in source
    assert "_context_requires_document_verification" not in source
    assert "ToolRuntime" in source
    assert "ModelReflector" in source
    assert "while True:" in source


def test_legacy_langgraph_runtime_is_not_reintroduced() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    runtime_dependencies = [str(item).lower() for item in pyproject["project"]["dependencies"]]
    adr = (ROOT / "docs/adr/004-langgraph.md").read_text(encoding="utf-8")

    assert not any(item.startswith("langgraph") for item in runtime_dependencies)
    assert not (ROOT / "src/rag_ops_guard/graph").exists()
    assert not (ROOT / "docs/diagrams/langgraph-flow.svg").exists()
    assert "Status: **Superseded**" in adr
