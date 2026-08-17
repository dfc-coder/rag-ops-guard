from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_canonical_make_targets_expose_one_local_pipeline() -> None:
    makefile = _text("Makefile")

    assert "\nup:\n" in makefile
    assert "$(MAKE) physical-ready" in makefile
    assert "\ndown: local-down\n" in makefile
    assert "\nui:\n" in makefile
    assert "scripts/run_chainlit_beta.sh" in makefile
    assert "\nchainlit-beta: ui\n" in makefile
    assert "\nbeta-react: ui\n" in makefile


def test_interactive_ui_uses_floci_api_instead_of_direct_agent() -> None:
    ui = _text("scripts/chainlit_api_ui.py")
    wrapper = _text("scripts/run_chainlit_beta.sh")

    assert "/v1/query" in ui
    assert "conversation_agent" not in ui
    assert "query_workflow" not in ui
    assert "rag_ops_guard.app" not in ui
    assert "chainlit_api_ui.py" in wrapper
    assert "Floci API Gateway -> Lambda -> Agent" in wrapper


def test_demo_uses_same_canonical_query_endpoint() -> None:
    demo = _text("scripts/demo.py")

    assert "/v1/query" in demo
    assert "query_workflow" not in demo
    assert "conversation_agent" not in demo
    assert "rag_ops_guard.app" not in demo
