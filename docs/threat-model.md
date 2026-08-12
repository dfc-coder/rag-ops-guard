# Threat Model — Beta 1

## Assets

- operational runbooks and incident knowledge;
- query inputs;
- generated recommendations;
- source provenance/citations;
- future production credentials (not present in the Beta corpus).

## Primary threats

### Unsupported operational recommendation

Mitigation: evidence-only prompt, deterministic resolver, abstention and golden tests.

### Indirect prompt injection in retrieved documents

Mitigation: evidence blocks are explicitly untrusted data; injected instructions never become system instructions; adversarial corpus/tests validate this boundary.

### Citation fabrication

Mitigation: generated citation IDs are checked against resolved evidence and unknown IDs reject the answer.

### Obsolete runbook wins semantic similarity

Mitigation: metadata status/version resolution occurs after semantic retrieval and before generation.

### Direct secret/policy bypass request

Mitigation: query-analysis route can terminate with `safety_blocked` before retrieval/generation.

### Accidental sensitive logging

Design rule: do not log full documents, full prompts or embedding vectors by default.

### Untrusted code on self-hosted release runner

Mitigation: real-model E2E workflow accepts only an internal `develop -> main` PR from `dfc-coder/rag-ops-guard`. Fork PRs must never execute on the trusted runner.
