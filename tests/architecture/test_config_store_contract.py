from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGSTORE = ROOT / "src/rag_ops_guard/configstore"


def test_config_store_never_scans_dynamodb() -> None:
    for path in CONFIGSTORE.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert ".scan(" not in source, path


def test_dynamo_partition_keys_come_from_scope_enum() -> None:
    source = (CONFIGSTORE / "dynamo_store.py").read_text(encoding="utf-8")
    assert "ConfigScope.GLOBAL.value" in source
    assert "ConfigScope.REGISTRY.value" in source
    assert '"PK": "GLOBAL"' not in source
    assert '"PK": "REGISTRY"' not in source
