from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
from typing import Any

import boto3

from rag_ops_guard.config import Settings
from rag_ops_guard.configstore.admin import (
    AdminPrincipal,
    ConfigAdminService,
    SecretMutationRequest,
    SecretMutationService,
)
from rag_ops_guard.configstore.admin_store import DynamoDbAdminStore
from rag_ops_guard.configstore.key_provider import KmsKeyProvider
from rag_ops_guard.configstore.phase3_secret_backend import EnvelopeSecretBackend
from rag_ops_guard.configstore.secret_service import EnvelopeSecretService
from rag_ops_guard.configstore.secret_store import DynamoDbSecretStore
from rag_ops_guard.configstore.tenant_store import TenantDynamoDbConfigStore

DEFAULT_CONFIG_TABLE = "rag-ops-config"
DEFAULT_KEK_REF = "alias/rag-ops-guard-config-secrets"


def _client_kwargs(settings: Settings) -> dict[str, object]:
    return {
        "endpoint_url": settings.aws_endpoint_url or None,
        "region_name": settings.aws_region,
        "aws_access_key_id": settings.aws_access_key_id,
        "aws_secret_access_key": settings.aws_secret_access_key.get_secret_value(),
    }


def _config_table() -> str:
    return os.environ.get("CONFIG_TABLE", DEFAULT_CONFIG_TABLE)


def _authenticated_principal(settings: Settings, tenant_id: str) -> AdminPrincipal:
    sts = boto3.client("sts", **_client_kwargs(settings))
    identity = sts.get_caller_identity()
    arn = str(identity.get("Arn") or "").strip()
    if not arn:
        raise RuntimeError("STS GetCallerIdentity did not return an authenticated ARN")
    return AdminPrincipal(principal_id=arn, tenant_ids=frozenset({tenant_id}))


def _config_store(settings: Settings, tenant_id: str) -> TenantDynamoDbConfigStore:
    return TenantDynamoDbConfigStore(
        endpoint_url=settings.aws_endpoint_url,
        region=settings.aws_region,
        access_key=settings.aws_access_key_id,
        secret_key=settings.aws_secret_access_key.get_secret_value(),
        table=_config_table(),
        tenant_id=tenant_id,
    )


def _admin_store(settings: Settings, tenant_id: str) -> DynamoDbAdminStore:
    return DynamoDbAdminStore(
        endpoint_url=settings.aws_endpoint_url,
        region=settings.aws_region,
        access_key=settings.aws_access_key_id,
        secret_key=settings.aws_secret_access_key.get_secret_value(),
        table=_config_table(),
        tenant_id=tenant_id,
    )


def _secret_backend(settings: Settings, request: SecretMutationRequest) -> EnvelopeSecretBackend:
    if not request.kek_ref:
        raise RuntimeError("secret mutation request has no KEK reference")
    resource = boto3.resource("dynamodb", **_client_kwargs(settings))
    kms = boto3.client("kms", **_client_kwargs(settings))
    envelope = EnvelopeSecretService(
        store=DynamoDbSecretStore(table=resource.Table(_config_table())),
        key_provider=KmsKeyProvider(client=kms),
    )
    return EnvelopeSecretBackend(service=envelope, kek_ref=request.kek_ref)


def _load_values(path: str) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration values file must contain a JSON object")
    return {str(name): value for name, value in payload.items()}


def _secret_from_prompt(*, confirmation: bool) -> bytes:
    first = getpass.getpass("Secret value: ").encode("utf-8")
    if not first:
        raise ValueError("secret value must not be empty")
    if confirmation:
        second = getpass.getpass("Confirm secret value: ").encode("utf-8")
        if first != second:
            raise ValueError("secret confirmation does not match")
    return first


def _services(settings: Settings) -> tuple[ConfigAdminService, SecretMutationService]:
    config = ConfigAdminService(store_factory=lambda tenant: _config_store(settings, tenant))
    secrets = SecretMutationService(
        store_factory=lambda tenant: _admin_store(settings, tenant),
        backend_factory=lambda request: _secret_backend(settings, request),
    )
    return config, secrets


def _print_status(service: ConfigAdminService, principal: AdminPrincipal, tenant_id: str) -> None:
    status = service.status(principal=principal, tenant_id=tenant_id)
    payload: dict[str, Any] = {
        "tenant_id": status.tenant_id,
        "revision_no": status.revision_no,
        "config_hash": status.config_hash,
        "revision_age_s": status.revision_age_s,
        "recent_audit": [
            {
                "actor": event.actor,
                "approving_principal": event.approving_principal,
                "action": event.action,
                "change_reason": event.change_reason,
                "hash_before": event.hash_before,
                "hash_after": event.hash_after,
                "source_revision": event.source_revision,
                "resulting_revision": event.resulting_revision,
                "secret_ref": event.secret_ref,
                "correlation_id": event.correlation_id,
                "timestamp": event.timestamp,
            }
            for event in status.recent_audit
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description="Authenticated Phase 5 configuration administration")
    parser.add_argument("--tenant-id", required=True)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show active revision and immutable administrative audit")

    publish = sub.add_parser("publish", help="Publish an append-only tenant config revision")
    publish.add_argument("--values", required=True, help="JSON object file with non-secret values")
    publish.add_argument("--reason", required=True)

    rollback = sub.add_parser("rollback", help="Create a new revision from historical values")
    rollback.add_argument("--revision", type=int, required=True)
    rollback.add_argument("--reason", required=True)

    request = sub.add_parser("secret-request", help="Create a dual-control secret mutation request")
    request.add_argument("--action", choices=("set", "rotate", "delete"), required=True)
    request.add_argument("--key", required=True)
    request.add_argument("--reason", required=True)
    request.add_argument("--kek-ref", default=DEFAULT_KEK_REF)

    approve = sub.add_parser("secret-approve", help="Approve and execute a pending secret mutation")
    approve.add_argument("--request-id", required=True)

    args = parser.parse_args()
    settings = Settings()
    principal = _authenticated_principal(settings, args.tenant_id)
    config_service, secret_service = _services(settings)

    if args.command == "status":
        _print_status(config_service, principal, args.tenant_id)
        return 0

    if args.command == "publish":
        published = config_service.publish(
            principal=principal,
            tenant_id=args.tenant_id,
            values=_load_values(args.values),
            change_reason=args.reason,
        )
        print(
            json.dumps(
                {
                    "tenant_id": args.tenant_id,
                    "revision_no": published.revision_no,
                    "config_hash": published.content_hash,
                },
                sort_keys=True,
            )
        )
        return 0

    if args.command == "rollback":
        published = config_service.rollback(
            principal=principal,
            tenant_id=args.tenant_id,
            revision_no=args.revision,
            change_reason=args.reason,
        )
        print(
            json.dumps(
                {
                    "tenant_id": args.tenant_id,
                    "revision_no": published.revision_no,
                    "config_hash": published.content_hash,
                    "rollback_from_revision": args.revision,
                },
                sort_keys=True,
            )
        )
        return 0

    if args.command == "secret-request":
        secret = None if args.action == "delete" else _secret_from_prompt(confirmation=True)
        pending = secret_service.request(
            principal=principal,
            tenant_id=args.tenant_id,
            action=args.action,
            key_name=args.key,
            change_reason=args.reason,
            secret=secret,
            kek_ref=None if args.action == "delete" else args.kek_ref,
        )
        print(
            json.dumps(
                {
                    "request_id": pending.request_id,
                    "tenant_id": pending.tenant_id,
                    "action": pending.action,
                    "key_name": pending.key_name,
                    "requested_by": pending.requested_by,
                    "status": pending.status,
                    "payload_sha256": pending.payload_sha256,
                },
                sort_keys=True,
            )
        )
        return 0

    if args.command == "secret-approve":
        store = _admin_store(settings, args.tenant_id)
        pending = store.get_secret_request(args.request_id)
        if pending is None:
            raise ValueError("secret mutation request does not exist")
        secret = None if pending.action == "delete" else _secret_from_prompt(confirmation=False)
        completed = secret_service.approve(
            principal=principal,
            tenant_id=args.tenant_id,
            request_id=args.request_id,
            secret=secret,
        )
        print(
            json.dumps(
                {
                    "request_id": completed.request_id,
                    "tenant_id": completed.tenant_id,
                    "action": completed.action,
                    "key_name": completed.key_name,
                    "requested_by": completed.requested_by,
                    "approved_by": completed.approved_by,
                    "status": completed.status,
                },
                sort_keys=True,
            )
        )
        return 0

    raise RuntimeError(f"unsupported command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
