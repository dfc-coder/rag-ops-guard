from __future__ import annotations

import argparse
import json
import os
import secrets
from typing import Any

import boto3

from rag_ops_guard.configstore.token_hashing import TenantTokenHasher
from rag_ops_guard.local_credentials import (
    LocalCredentialError,
    SecretServiceCredentialStore,
    TenantApiCredential,
)
from rag_ops_guard.tenancy import KeyLayout

DEFAULT_ENDPOINT = "http://localhost:4566"
DEFAULT_REGION = "us-east-1"
DEFAULT_STACK = "RagOpsGuardLocal"


def _client(service: str) -> Any:
    return boto3.client(
        service,
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL", DEFAULT_ENDPOINT),
        region_name=os.environ.get("AWS_REGION", DEFAULT_REGION),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "test"),
    )


def _stack_outputs() -> dict[str, str]:
    stack_name = os.environ.get("RAG_OPS_CDK_STACK_NAME", DEFAULT_STACK)
    response = _client("cloudformation").describe_stacks(StackName=stack_name)
    stacks = response.get("Stacks", [])
    if len(stacks) != 1:
        raise RuntimeError(f"expected one CloudFormation stack named {stack_name!r}")
    return {
        str(item["OutputKey"]): str(item["OutputValue"])
        for item in stacks[0].get("Outputs", [])
        if item.get("OutputKey") and item.get("OutputValue")
    }


def _tenant_table() -> str:
    value = _stack_outputs().get("TenantTableName", "").strip()
    if not value:
        raise RuntimeError("CDK stack output 'TenantTableName' is missing; run `make up` first")
    return value


def _normalize_tenant(tenant_id: str) -> str:
    return KeyLayout(tenant_id).tenant_id


def _record(key_id: str) -> dict[str, Any] | None:
    response = _client("dynamodb").get_item(
        TableName=_tenant_table(),
        Key={"key_id": {"S": key_id}},
        ConsistentRead=True,
    )
    item = response.get("Item")
    return item if isinstance(item, dict) else None


def _item_tenant(item: dict[str, Any]) -> str:
    raw = item.get("tenant_id", {})
    return str(raw.get("S") or "") if isinstance(raw, dict) else ""


def _item_enabled(item: dict[str, Any]) -> bool:
    raw = item.get("enabled", {})
    return bool(raw.get("BOOL")) if isinstance(raw, dict) else False


def _item_hash(item: dict[str, Any]) -> str:
    raw = item.get("token_hash", {})
    return str(raw.get("S") or "") if isinstance(raw, dict) else ""


def _ensure_key_id_available(*, tenant_id: str, key_id: str, allow_existing: bool) -> None:
    item = _record(key_id)
    if item is None:
        return
    existing_tenant = _item_tenant(item)
    if existing_tenant != tenant_id:
        raise RuntimeError(
            f"key_id {key_id!r} already belongs to tenant {existing_tenant!r}; choose another key id"
        )
    if not allow_existing:
        raise RuntimeError(
            f"credential {key_id!r} already exists for tenant {tenant_id!r}; use tenant-rotate"
        )


def _register(*, tenant_id: str, credential: TenantApiCredential) -> None:
    _ensure_key_id_available(tenant_id=tenant_id, key_id=credential.key_id, allow_existing=True)
    _client("dynamodb").put_item(
        TableName=_tenant_table(),
        Item={
            "key_id": {"S": credential.key_id},
            "tenant_id": {"S": tenant_id},
            "token_hash": {"S": TenantTokenHasher().hash_token(credential.secret)},
            "enabled": {"BOOL": True},
        },
    )


def _create(*, tenant_id: str, key_id: str, store: SecretServiceCredentialStore) -> None:
    if store.get_api_key(tenant_id) is not None:
        raise RuntimeError(
            f"a local credential already exists for tenant {tenant_id!r}; use tenant-sync or tenant-rotate"
        )
    _ensure_key_id_available(tenant_id=tenant_id, key_id=key_id, allow_existing=False)
    credential = TenantApiCredential(key_id=key_id, secret=secrets.token_urlsafe(32))
    store.set_api_key(tenant_id, credential.api_key)
    try:
        _register(tenant_id=tenant_id, credential=credential)
    except Exception:
        store.delete_api_key(tenant_id)
        raise
    print(
        json.dumps(
            {
                "tenant_id": tenant_id,
                "key_id": key_id,
                "enabled": True,
                "local_store": "Secret Service",
                "server_plaintext_persisted": False,
                "status": "created",
            },
            sort_keys=True,
        )
    )


def _sync(*, tenant_id: str, store: SecretServiceCredentialStore) -> None:
    credential = TenantApiCredential.parse(store.require_api_key(tenant_id))
    _register(tenant_id=tenant_id, credential=credential)
    print(
        json.dumps(
            {
                "tenant_id": tenant_id,
                "key_id": credential.key_id,
                "enabled": True,
                "status": "synchronized",
            },
            sort_keys=True,
        )
    )


def _rotate(*, tenant_id: str, store: SecretServiceCredentialStore) -> None:
    previous = TenantApiCredential.parse(store.require_api_key(tenant_id))
    _ensure_key_id_available(tenant_id=tenant_id, key_id=previous.key_id, allow_existing=True)
    replacement = TenantApiCredential(
        key_id=previous.key_id,
        secret=secrets.token_urlsafe(32),
    )
    store.set_api_key(tenant_id, replacement.api_key)
    try:
        _register(tenant_id=tenant_id, credential=replacement)
    except Exception:
        store.set_api_key(tenant_id, previous.api_key)
        raise
    print(
        json.dumps(
            {
                "tenant_id": tenant_id,
                "key_id": replacement.key_id,
                "enabled": True,
                "status": "rotated",
            },
            sort_keys=True,
        )
    )


def _revoke(*, tenant_id: str, key_id: str, store: SecretServiceCredentialStore) -> None:
    local = store.get_api_key(tenant_id)
    if local is not None:
        key_id = TenantApiCredential.parse(local).key_id
    item = _record(key_id)
    if item is not None:
        existing_tenant = _item_tenant(item)
        if existing_tenant != tenant_id:
            raise RuntimeError(
                f"key_id {key_id!r} belongs to tenant {existing_tenant!r}, not {tenant_id!r}"
            )
        disabled = dict(item)
        disabled["enabled"] = {"BOOL": False}
        _client("dynamodb").put_item(TableName=_tenant_table(), Item=disabled)
    store.delete_api_key(tenant_id)
    print(
        json.dumps(
            {
                "tenant_id": tenant_id,
                "key_id": key_id,
                "enabled": False,
                "local_credential_present": False,
                "status": "revoked",
            },
            sort_keys=True,
        )
    )


def _status(*, tenant_id: str, key_id: str, store: SecretServiceCredentialStore) -> int:
    local = store.get_api_key(tenant_id)
    credential = TenantApiCredential.parse(local) if local is not None else None
    if credential is not None:
        key_id = credential.key_id
    item = _record(key_id)
    enabled = item is not None and _item_enabled(item)
    tenant_matches = item is not None and _item_tenant(item) == tenant_id
    verifier_matches = bool(
        credential is not None
        and item is not None
        and tenant_matches
        and TenantTokenHasher().verify_token(
            encoded_hash=_item_hash(item),
            token=credential.secret,
        )
    )
    ready = bool(local and enabled and tenant_matches and verifier_matches)
    print(
        json.dumps(
            {
                "tenant_id": tenant_id,
                "key_id": key_id,
                "local_credential_present": local is not None,
                "server_record_present": item is not None,
                "enabled": enabled,
                "tenant_matches": tenant_matches,
                "verifier_matches": verifier_matches,
                "ready": ready,
            },
            sort_keys=True,
        )
    )
    return 0 if ready else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage local tenant API credentials outside infrastructure provisioning"
    )
    parser.add_argument("command", choices=("create", "sync", "rotate", "revoke", "status"))
    parser.add_argument("--tenant-id", default="default")
    parser.add_argument("--key-id")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    tenant_id = _normalize_tenant(str(args.tenant_id))
    key_id = str(args.key_id or tenant_id).strip()
    if not key_id or "." in key_id:
        raise ValueError("key id must be non-empty and must not contain '.'")
    store = SecretServiceCredentialStore()

    if args.command == "create":
        _create(tenant_id=tenant_id, key_id=key_id, store=store)
        return 0
    if args.command == "sync":
        _sync(tenant_id=tenant_id, store=store)
        return 0
    if args.command == "rotate":
        _rotate(tenant_id=tenant_id, store=store)
        return 0
    if args.command == "revoke":
        _revoke(tenant_id=tenant_id, key_id=key_id, store=store)
        return 0
    if args.command == "status":
        return _status(tenant_id=tenant_id, key_id=key_id, store=store)
    raise RuntimeError(f"unsupported command {args.command!r}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LocalCredentialError as exc:
        raise SystemExit(str(exc)) from exc
