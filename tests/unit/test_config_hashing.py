from pydantic import SecretStr

from rag_ops_guard.configstore.hashing import canonical_config_json, content_hash


def test_hash_is_stable_across_key_order() -> None:
    left = {"b": 2, "a": 1}
    right = {"a": 1, "b": 2}
    assert canonical_config_json(left) == canonical_config_json(right)
    assert content_hash(left) == content_hash(right)


def test_hash_is_64_hex_characters() -> None:
    digest = content_hash({"value": 1})
    assert len(digest) == 64
    int(digest, 16)


def test_hash_changes_when_any_value_changes() -> None:
    assert content_hash({"a": 1, "b": 2}) != content_hash({"a": 1, "b": 3})


def test_secret_is_represented_by_its_own_sha256_not_plaintext() -> None:
    secret = "super-secret-value"
    payload = canonical_config_json({"secret": SecretStr(secret)})
    assert secret.encode() not in payload
    assert content_hash({"secret": SecretStr(secret)}) == content_hash(
        {"secret": SecretStr(secret)}
    )
