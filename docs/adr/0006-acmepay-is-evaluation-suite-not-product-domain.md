# ADR-0006: AcmePay is the evaluation suite, not the product domain

## Status

Accepted for the generic-agent pivot.

## Context

The repository grew around a synthetic-but-realistic AcmePay Integration Ops corpus. Over time, operational concepts leaked from evaluation fixtures into product behavior:

- document metadata became mandatory and domain-specific;
- retrieval/resolution logic assumed operational concepts such as environment, authority, version and supersession;
- calibration and routing decisions were interpreted as product scope;
- AcmePay vocabulary risked becoming control-flow heuristics rather than test data.

That coupling blocks the target product: a generic conversational ReAct agent that can ingest arbitrary user documents and use document search as one optional tool.

The existing AcmePay assets are still valuable. They contain grounded, unsupported, adversarial and multi-turn cases that are useful for regression testing.

## Decision

AcmePay is retained as the canonical evaluation suite, not as the product domain.

The product must not require a document to be an Integration Ops artifact in order to ingest, retrieve or cite it.

The first generic-ingestion scope is intentionally limited to:

```text
.md
.txt
```

A plain Markdown or text document without YAML front matter must be ingestible.

The generic normalized document requires only generic identity/content fields:

```text
id
title
source
mime_type
content
metadata (optional map)
```

Rich operational metadata remains optional. When present and valid, it may enable governance policy such as:

- version selection;
- authority preference;
- effective-date handling;
- environment filtering;
- supersession handling.

When absent, those policies are skipped rather than making the document invalid.

Front matter that is present but malformed is an ingestion error. The system must not silently reinterpret malformed metadata as a generic document.

PDF, DOCX and HTML ingestion are deferred to later isolated units. They are not part of the first generic-ingestion release.

## Evaluation role

The existing AcmePay corpus and datasets remain responsible for regression coverage including:

- grounded questions;
- in-domain-but-unanswerable questions;
- out-of-domain/general questions;
- citation integrity;
- adversarial prompt/secret-extraction behavior;
- multi-turn grounded follow-up;
- retrieval calibration.

The three retrieval-calibration classes remain semantically distinct:

```text
grounded
in_domain_unanswerable
out_of_domain
```

They are test labels. They do not define which subjects the product is allowed to discuss.

## Alternatives considered

### Keep required operational metadata and synthesize it for generic documents

Rejected. Fabricated metadata hides the domain coupling instead of removing it.

### Remove AcmePay entirely

Rejected. It is useful, deterministic regression material and already exercises important grounding/safety behavior.

### Add PDF/DOCX/HTML immediately

Rejected for the first pivot release. Multiple extractors, chunking differences and metadata normalization would enlarge U2 before the generic document contract is proven.

## Consequences

Positive:

- the same agent can ingest notes, manuals, specifications, contracts or other text documents without pretending they are runbooks;
- existing Ops governance can survive as an optional policy layer;
- AcmePay remains useful without constraining product scope;
- future document formats can be added behind the same normalized document contract.

Costs:

- ingestion and resolver code must support documents without operational metadata;
- domain-specific retrieval heuristics must be removed or isolated;
- current golden datasets need status/contract realignment after the segmented-response migration.

## Migration impact

U2 implements generic Markdown/TXT ingestion and optional governance metadata.

U4 calibrates relevance/evidence behavior without turning AcmePay labels into a global product scope classifier.

U6 realigns the golden dataset to the new response contract.

U7 removes legacy pipelines and remaining dead/domain-coupled control paths.
