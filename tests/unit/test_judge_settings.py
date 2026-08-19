from pathlib import Path

from rag_ops_guard.config import Settings
from rag_ops_guard.evaluation.judge import judge_connection_from_env, judge_identity_from_env


def test_judge_identity_uses_validated_settings_model() -> None:
    identity = judge_identity_from_env()
    assert identity.model == Settings().resolved_ragas_judge_model


def test_local_judge_without_explicit_key_uses_local_sentinel(monkeypatch) -> None:
    monkeypatch.delenv("RAGAS_JUDGE_API_KEY", raising=False)
    connection = judge_connection_from_env()
    assert connection.api_key == "local"


def test_judge_has_no_shadow_ragas_environment_reads() -> None:
    source = Path("src/rag_ops_guard/evaluation/judge.py").read_text(encoding="utf-8")
    assert 'os.environ["RAGAS_' not in source
    assert 'os.environ.get("RAGAS_' not in source
