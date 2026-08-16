from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_final_demo_targets_use_canonical_agent_and_qwen3_4b() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    compose = (ROOT / "docker/docker-compose.yml").read_text(encoding="utf-8")

    assert "beta-react:" in makefile
    assert "chainlit-gate:" in makefile
    assert "eval-judge-calibrate:" in makefile
    assert "MODEL_FILES=Qwen3-4B-Q4_K_M.gguf" in makefile
    assert "Qwen3-4B-Q4_K_M.gguf" in compose
    assert "qwen3-4b-rag" in compose


def test_obsolete_react_smoke_shims_are_not_release_targets() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    for obsolete in (
        "react-smoke:",
        "react-rag-smoke:",
        "react-direct-stream-smoke:",
        "react-direct-ui-smoke:",
    ):
        assert obsolete not in makefile


def test_release_check_keeps_style_advisory_but_correctness_blocking() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "release-check: lint-advisory types test test-integration test-e2e eval" in makefile
    assert "lint-advisory:" in makefile
    assert "types:" in makefile
    assert "test-unit:" in makefile
