# Local tenant credentials

Local tenant API credentials are client credentials. They are not infrastructure bootstrap configuration and MUST NOT be supplied through `.env`, shell exports, CDK, CloudFormation, Lambda environment variables, or plaintext DynamoDB attributes.

## Storage model

```text
Fedora workstation                         Floci / AWS-compatible server
------------------                         -----------------------------
Secret Service / GNOME Keyring             rag-ops-tenants (DynamoDB)
  key_id.secret                              key_id
        |                                    tenant_id
        |                                    Argon2id token_hash
        +---------- authenticated request --> enabled
```

The cleartext local credential is stored in the desktop Secret Service. The server stores only the Argon2id verifier defined by the Phase 4 authentication contract.

`secret-tool store` receives the credential through stdin, not through command-line arguments. Local credential commands never print the credential value.

## Lifecycle

Infrastructure has no tenant-credential prerequisite:

```bash
make local-clean
make up
```

Create the first local credential explicitly:

```bash
make tenant-create TENANT=default
make tenant-status TENANT=default
```

`tenant-create` generates a cryptographically random secret, stores `key_id.secret` in Secret Service, and writes only the Argon2id verifier to the tenant table.

After `make local-clean`, the OS credential intentionally survives while Floci state is removed. Restore only the verifier with:

```bash
make up
make tenant-sync TENANT=default
make tenant-status TENANT=default
```

Rotate or revoke explicitly:

```bash
make tenant-rotate TENANT=default
make tenant-revoke TENANT=default
```

Rotation keeps the same key id and generates a new random secret. Revocation disables the server record and removes the local Secret Service entry.

## Authenticated readiness

Once a tenant credential exists or has been synchronized:

```bash
make ready TENANT=default
make connectivity TENANT=default
```

Canonical local clients (`connectivity`, corpus ingestion, demo, Chainlit and Golden evaluation) resolve the API credential from Secret Service rather than `RAG_OPS_API_KEY`.

## Fedora prerequisite

The workstation must provide the freedesktop Secret Service CLI `secret-tool`. Check with:

```bash
command -v secret-tool
```

If the command is unavailable, install the Fedora package that provides `secret-tool`/libsecret before tenant onboarding.

## Security invariants

- `make up` MUST succeed with zero tenants and without an API-key environment variable.
- Tenant credential creation is an explicit administrative action.
- Cleartext tenant credentials MUST NOT be persisted by the server.
- Cleartext tenant credentials MUST NOT be written to repository files or `.env`.
- Cleartext tenant credentials MUST NOT appear in process arguments or command output.
- `tenant-sync` may re-register an Argon2id verifier from the OS credential after local infrastructure is destroyed.
- Runtime authentication continues to derive `tenant_id` from the verified credential record, never from request-body authority.

This is a Phase 5 local/operator lifecycle improvement. It does not implement the Phase 6 environmentless runtime control plane; the remaining non-secret/bootstrap environment contract is removed in Phase 6.
