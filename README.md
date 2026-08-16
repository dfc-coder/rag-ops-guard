# RAG Ops Guard

**Local-first conversational ReAct agent with extensible tools and auditable document grounding.**

This README is the product specification for the generic-agent pivot. During the migration, implementation may temporarily lag this specification; each work unit closes one explicit gap through SDD/TDD.

## Product definition

RAG Ops Guard is a generic conversational agent. It can answer normally from the generation model, call tools when useful, and use ingested documents as an optional source of grounded evidence.

The product is **not** an Integration Ops-only copilot. The AcmePay operational corpus remains in the repository as an evaluation suite, not as the definition of what questions the product may answer.

### Product invariant

> No claim may be presented as grounded unless it is supported by evidence admitted in that turn.

The previous global invariant — "insufficient document evidence means no answer" — is retired. A response may contain grounded and ungrounded claims together, but that distinction must be explicit and machine-verifiable at claim/segment level.

## Target runtime architecture

```text
Chainlit ---------\
REST /v1/query ----+--> ConversationAgent --> ReAct loop --> Tool registry
CLI / future UI --/                               |
                                                   +--> search_documents
                                                   +--> list_documents
                                                   +--> future tools
```

`ConversationAgent` is the single application-level owner of conversation, memory, streaming and tool orchestration. Client adapters must not create independent routing or agent pipelines.

Deterministic controls remain outside or around model choice where correctness requires them:

- safety before model/tool execution;
- authorization/document access when introduced;
- citation integrity;
- evidence admission and grounding policy;
- transactional conversation-state correctness.

RAG is one capability of the agent, not the center of the architecture.

## Public response contract

A response is an ordered sequence of segments. A segment is either:

- **grounded** — contains at least one citation to evidence admitted in the current turn;
- **ungrounded** — contains no citations and is presented as model/general/tool output rather than document-backed fact.

The public status is derived from those segments, not assigned independently.

| Status | Condition | Citations |
| --- | --- | --- |
| `answered_grounded` | one or more segments, all grounded | one or more |
| `answered_mixed` | at least one grounded and one ungrounded segment | one or more |
| `answered_ungrounded` | one or more segments, none grounded | zero |
| `clarification_required` | no answer segments; a clarification question is required | zero |
| `safety_blocked` | deterministic safety blocks the turn | zero |
| `error` | the turn cannot complete safely | zero |

`insufficient_evidence` is not a final public answer state in the pivot design. Missing document support may be reported in an ungrounded segment, while any document-backed claims in the same answer remain individually grounded.

`QueryResponse.citations` is a derived union of citations attached to grounded segments. An ungrounded segment carrying a citation, or a citation to a chunk not admitted in that turn, is invalid by contract.

## Tool model

The agent chooses tools through the ReAct loop. The initial document tools are conceptually:

```text
search_documents
list_documents
```

Document search performs retrieval and returns typed evidence/search data. It does **not** decide whether a generated claim is grounded. Grounding is a separate policy/integrity decision over admitted evidence.

The architecture must remain open to additional tools such as APIs, databases, MCP servers, calculators, incident systems and business workflows without changing the conversation core.

## Generic document ingestion

The first generic-ingestion release intentionally supports only:

```text
.md
.txt
```

PDF, DOCX and HTML are deferred until the normalized document model and generic chunking path are stable. Adding another extractor is a separate work unit, not part of the first pivot release.

### Metadata contract

A plain Markdown or text document must be ingestible without YAML front matter.

The normalized document requires only generic identity/content fields such as:

```text
id
title
source
mime_type
content
metadata (optional map)
```

When rich operational metadata is present and valid, it is preserved and may enable optional governance policy such as version, authority, effective-date, environment or supersession handling. Absence of that metadata must not make a generic document invalid.

If front matter is present but malformed, ingestion fails loudly rather than silently downgrading the document to generic mode.

## Corpus and evaluation

The synthetic AcmePay corpus under `knowledge-base/` is the regression/evaluation suite.

It is used to preserve:

- golden grounded cases;
- in-domain-but-unanswerable cases;
- out-of-domain/general cases;
- adversarial cases;
- retrieval calibration;
- citation and grounding integrity.

AcmePay vocabulary, Integration Ops concepts and current fixtures must not become control-flow rules or mandatory document metadata in the generic product.

## Security contract

User-ingested and retrieved document text is untrusted data.

The product must preserve these guarantees:

- direct secret-extraction attacks are blocked before generation/tool execution when deterministic safety can identify them;
- instructions embedded inside documents are never treated as agent instructions;
- a malicious document may still be retrieved/cited as data without its embedded instructions being executed;
- document content cannot cause citation or disclosure of non-admitted evidence;
- adversarial tests must exercise the canonical `ConversationAgent` path, not historical/dead prompts.

## Relevance and grounding direction

Retrieval relevance and evidence support are separate signals.

The evaluation dataset distinguishes:

```text
grounded
in_domain_unanswerable
out_of_domain
```

Thresholds must come from measured calibration on the target runtime. If score distributions overlap so strongly that a threshold cannot satisfy the declared correctness/recall requirements, the gate must fail and the retrieval design must be revisited; thresholds are not adjusted by intuition to make a release pass.

The failed `agent/beta-unified-react-core` physical gate is retained as evidence for this rule: repository CI alone is not sufficient for a client freeze.

## SDD/TDD development contract

The pivot follows Spec-Driven Development plus Test-Driven Development:

1. specification/ADR first;
2. test that demonstrates the missing behavior and fails;
3. minimum implementation to make it pass;
4. refactor without behavior change;
5. CI gate or architecture fitness function where the rule can regress structurally.

Each specification rule receives a stable `SPEC-<unit>.<n>` identifier when its implementation unit starts. Tests cite that identifier.

No `skip` or `xfail` is used to make a unit green. Diagnostic output belongs in PR discussion, not committed diagnostic files/workflows.

Work units are merged independently into the stable pivot integration branch; they are not stacked PR-on-PR chains.

## Pivot integration point

The abandoned beta-freeze head is preserved as a technical checkpoint:

```text
716f3f46e81c5bfdd91912a4f79d88bcd59ec5ce
```

Stable pivot integration branch:

```text
pivot/generic-react
```

That checkpoint contributes the useful single-`ConversationAgent` work, but it is **not** a client release. The generic pivot must satisfy its own product specs and physical evaluation gates before it can merge back into `develop` or be frozen for a client.

## Planned work units

```text
U0   product specification + ADRs
U1a  tool/model ports and adapters
U1   segmented response contract
U3   direct + indirect prompt-injection security
U2   generic Markdown/TXT ingestion
U4   calibrated relevance/evidence floors
U5   RAGAS realignment and per-case gates
U6   evaluation dataset/status realignment
U7   removal of legacy/dead pipelines
```

U0 is specification-only. No runtime behavior changes are part of this unit.

## Runtime implementation currently retained

The existing local runtime remains useful during migration:

```text
CPU
├── Python orchestration
├── Qwen 3.5 2B generation / llama.cpp
├── BM25 / RRF
└── Floci

Intel Iris Xe / OpenVINO Model Server
├── Qwen3 Embedding 0.6B
└── Qwen3 Reranker 0.6B
```

Hardware/backend choices are implementation details and may evolve independently from the product contract above.

## Repository structure

```text
src/rag_ops_guard/       application and agent code
knowledge-base/          AcmePay evaluation corpus
evaluation/              golden/adversarial/calibration datasets and evaluators
tests/                   unit, property, architecture, integration and E2E tests
infra/cdk/               AWS target architecture
docker/                  local container topology
scripts/                 local runtime, provisioning and smoke commands
docs/                    architecture, ADRs and threat model
.github/workflows/        CI and release validation
```

## Engineering documents

- [Architecture](docs/architecture.md)
- [Technology stack](docs/technology-stack.md)
- [Evaluation strategy](docs/evaluation-strategy.md)
- [Threat model](docs/threat-model.md)
- [Architecture Decision Records](docs/adr/)

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
