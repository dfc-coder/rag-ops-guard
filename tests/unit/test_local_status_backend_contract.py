from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_local_status_detects_lambda_openvino_backend_drift() -> None:
    ops = (ROOT / "scripts/local/ops.py").read_text(encoding="utf-8")

    assert '"EMBEDDING_BASE_URL"' in ops
    assert '"RERANKER_BASE_URL"' in ops
    assert 'OVMS_CONTAINER_NAME' in ops
    assert 'Lambda RAG' in ops
    assert 'backend_mismatch' in ops
