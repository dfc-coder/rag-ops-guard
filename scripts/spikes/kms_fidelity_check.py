#!/usr/bin/env python3
"""Phase 3 entry spike: measure the KMS surface available through Floci.

This script is intentionally standalone and uses only boto3, which is already a
runtime dependency. It does not decide the production design; it reports whether
Floci faithfully supports the minimum KMS operations required by the Phase 3
KeyProvider contract.

Required fidelity path:
    CreateKey
    -> GenerateDataKey(AES_256 / 32 bytes)
    -> Encrypt/Decrypt roundtrip
    -> re-wrap the generated DEK under a second KEK

No plaintext key material is printed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

import boto3


@dataclass(frozen=True)
class ProbeResult:
    create_key: bool
    generate_data_key: bool
    encrypt_decrypt: bool
    rewrap: bool

    @property
    def ok(self) -> bool:
        return all(
            (
                self.create_key,
                self.generate_data_key,
                self.encrypt_decrypt,
                self.rewrap,
            )
        )

    def as_dict(self) -> dict[str, bool]:
        return {
            "create_key": self.create_key,
            "generate_data_key": self.generate_data_key,
            "encrypt_decrypt": self.encrypt_decrypt,
            "rewrap": self.rewrap,
            "ok": self.ok,
        }


def _kms_client(
    *, endpoint_url: str, region: str, access_key: str, secret_key: str
) -> Any:
    return boto3.client(
        "kms",
        endpoint_url=endpoint_url,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def _key_id(response: dict[str, Any]) -> str:
    metadata = response.get("KeyMetadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("CreateKey response did not contain KeyMetadata")
    key_id = metadata.get("KeyId")
    if not isinstance(key_id, str) or not key_id:
        raise RuntimeError("CreateKey response did not contain a usable KeyId")
    return key_id


def _best_effort_cleanup(client: Any, key_ids: list[str]) -> None:
    for key_id in key_ids:
        try:
            client.schedule_key_deletion(KeyId=key_id, PendingWindowInDays=7)
        except Exception:
            # Cleanup fidelity is not part of the Phase 3 entry decision.
            pass


def run_probe(client: Any) -> ProbeResult:
    created: list[str] = []
    create_key_ok = False
    generate_data_key_ok = False
    encrypt_decrypt_ok = False
    rewrap_ok = False

    try:
        kek_v1 = _key_id(
            client.create_key(Description="rag-ops-guard phase3 fidelity kek-v1")
        )
        created.append(kek_v1)
        kek_v2 = _key_id(
            client.create_key(Description="rag-ops-guard phase3 fidelity kek-v2")
        )
        created.append(kek_v2)
        create_key_ok = True

        generated = client.generate_data_key(KeyId=kek_v1, KeySpec="AES_256")
        dek_plaintext = generated.get("Plaintext")
        dek_wrapped_v1 = generated.get("CiphertextBlob")
        if not isinstance(dek_plaintext, bytes) or len(dek_plaintext) != 32:
            raise RuntimeError("GenerateDataKey did not return a 32-byte plaintext DEK")
        if not isinstance(dek_wrapped_v1, bytes) or not dek_wrapped_v1:
            raise RuntimeError("GenerateDataKey did not return CiphertextBlob")
        generate_data_key_ok = True

        probe_payload = b"rag-ops-guard-kms-fidelity-roundtrip"
        encrypted = client.encrypt(KeyId=kek_v1, Plaintext=probe_payload)
        ciphertext = encrypted.get("CiphertextBlob")
        if not isinstance(ciphertext, bytes) or not ciphertext:
            raise RuntimeError("Encrypt did not return CiphertextBlob")
        decrypted = client.decrypt(CiphertextBlob=ciphertext).get("Plaintext")
        if decrypted != probe_payload:
            raise RuntimeError("Encrypt/Decrypt plaintext roundtrip mismatch")
        encrypt_decrypt_ok = True

        unwrapped_v1 = client.decrypt(CiphertextBlob=dek_wrapped_v1).get("Plaintext")
        if unwrapped_v1 != dek_plaintext:
            raise RuntimeError("Generated DEK could not be unwrapped under KEK v1")

        wrapped_v2 = client.encrypt(KeyId=kek_v2, Plaintext=unwrapped_v1).get(
            "CiphertextBlob"
        )
        if not isinstance(wrapped_v2, bytes) or not wrapped_v2:
            raise RuntimeError("Re-wrap under KEK v2 did not return CiphertextBlob")
        if wrapped_v2 == dek_wrapped_v1:
            raise RuntimeError("Re-wrap returned the original wrapped DEK unchanged")

        unwrapped_v2 = client.decrypt(CiphertextBlob=wrapped_v2).get("Plaintext")
        if unwrapped_v2 != dek_plaintext:
            raise RuntimeError("Re-wrapped DEK did not roundtrip under KEK v2")
        rewrap_ok = True

        return ProbeResult(
            create_key=create_key_ok,
            generate_data_key=generate_data_key_ok,
            encrypt_decrypt=encrypt_decrypt_ok,
            rewrap=rewrap_ok,
        )
    finally:
        _best_effort_cleanup(client, created)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-url", default="http://localhost:4566")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--access-key", default="test")
    parser.add_argument("--secret-key", default="test")
    args = parser.parse_args()

    client = _kms_client(
        endpoint_url=args.endpoint_url,
        region=args.region,
        access_key=args.access_key,
        secret_key=args.secret_key,
    )

    try:
        result = run_probe(client)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "endpoint": args.endpoint_url,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "decision": "LOCAL_KEY_PROVIDER_ONLY",
                },
                sort_keys=True,
            )
        )
        return 1

    payload: dict[str, object] = {
        **result.as_dict(),
        "endpoint": args.endpoint_url,
        "decision": "KMS_FIDELITY_SUFFICIENT" if result.ok else "LOCAL_KEY_PROVIDER_ONLY",
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
