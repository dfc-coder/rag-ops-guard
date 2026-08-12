# Security Exceptions — Beta 1 Evaluation Tooling

The runtime/dev dependency set is audited without exceptions. Evaluation-only dependencies are audited separately because two current advisories have no patched upstream release as of 2026-08-12.

## PYSEC-2026-3046 — `ragas 0.4.3`

- Scope: optional `eval` dependency only.
- Runtime query/ingestion path: not imported.
- Exposure: trusted release runner, synthetic AcmePay corpus, local models.
- Upstream state: `0.4.3` is the latest available RAGAS release at the time this exception was recorded; `pip-audit` provides no fixed version.
- Mitigation: exception is limited to this advisory ID; all other evaluation dependency vulnerabilities remain blocking.
- Removal condition: delete the exception as soon as a patched RAGAS release is available and validated.

## PYSEC-2026-2447 — `diskcache 5.6.3`

- Scope: transitive evaluation-tooling dependency; not an explicit runtime dependency.
- Runtime query/ingestion path: not imported.
- Exposure: trusted release runner only.
- Upstream state: `5.6.3` is the latest available DiskCache release at the time this exception was recorded; `pip-audit` provides no fixed version.
- Mitigation: exception is limited to this advisory ID; any additional DiskCache advisory still blocks CI.
- Removal condition: delete the exception once a patched upstream/transitive dependency path is available.

## Policy

Exceptions are advisory-ID specific, documented and temporary. They must never be replaced with a blanket `pip-audit` disable, severity suppression, or `|| true` in the blocking CI gate.
