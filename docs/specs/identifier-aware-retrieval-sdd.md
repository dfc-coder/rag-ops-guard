# Identifier-Aware Retrieval SDD

## Problem

The demo must distinguish a near-domain query about an undocumented named system from a supported query about a documented system. Semantic similarity alone is insufficient: `Cual es el timeout exacto de SAP en produccion?` can rank Calypso timeout documents highly because the operational vocabulary is similar.

Trusted conversational focus creates a second risk: a self-contained topic switch can be rewritten using the previous supported topic, producing false support.

Identifier extraction itself must also avoid false anchors caused only by sentence capitalization. For example, in `Se puede hacer bulk replay del payment DLQ?`, `Se` is ordinary sentence-initial Spanish and must not suppress the real technical anchor `DLQ`.

## Design

1. Candidate generation remains Dense + BM25 -> RRF -> EvidenceResolver.
2. Before cross-encoder reranking, the runtime extracts explicit identifiers from the user query using surface-form rules only: acronyms, CamelCase, non-initial title-case technical names, and identifiers containing digits.
3. Plain Title Case in the first token of a sentence is not an identifier signal by itself. Acronyms, CamelCase and digit-bearing identifiers remain valid in any position.
4. The detector is generic and contains no product/system allowlist.
5. Generic protocol/category codes such as API, SLA, DLQ and P1 are not preferred over a co-occurring named identifier. When no named identifier exists, the generic technical identifier remains a valid anchor.
6. When a query contains a named identifier, a resolved candidate must contain at least one exact normalized identifier token in its title, metadata or chunk text before it can be reranked/admitted.
7. When no explicit identifier is present, retrieval behavior is unchanged.
8. Trusted follow-up rewriting is permitted only when the current query has no explicit new named topic, or its identifier is present in the prior grounded query/source titles.
9. A topic switch that names an unsupported system must not inherit evidence from the previous topic. It must remain unsupported and clear focus after the turn.
10. Retrieval calibration must include operational hard negatives containing domain vocabulary but unsupported named systems, in addition to broad out-of-domain negatives.

## Required examples

- `Cuantos reintentos permite Calypso?` -> `Payment Retry Policy` remains admissible because the evidence explicitly mentions Calypso.
- `que pasa con SendGrid?` -> SendGrid evidence remains admissible.
- `Se puede hacer bulk replay del payment DLQ?` -> `DLQ` remains the technical anchor; sentence-initial `Se` must not filter the candidate set.
- `Y despues del tercero?` after grounded Calypso retries -> trusted rewrite is allowed because the turn is elliptical and introduces no new named identifier.
- `Cual es el timeout exacto de SAP en produccion?` after grounded Calypso retries -> Calypso timeout evidence is not admissible; trusted rewrite must not cross the SAP topic boundary.
- `Como hago replay del DLQ de Kafka?` -> a payment DLQ document is not enough merely because both mention DLQ; Kafka is the named anchor.

## TDD acceptance

Unit tests must prove:

- generic identifier extraction for SAP, SendGrid and Calypso;
- sentence-initial Title Case does not become a false entity anchor;
- a standalone generic technical identifier such as DLQ remains usable when there is no named system anchor;
- elliptical follow-ups remain rewriteable;
- named topic switches are not rewritten with prior focus;
- a high-scoring Calypso timeout candidate cannot support a SAP timeout query;
- a legitimate cross-language Calypso retry query still admits `Payment Retry Policy`;
- hard-negative calibration includes unsupported operational systems.

The real acceptance gate remains `make demo-ready` on the client runtime. No score threshold change is allowed to compensate for an identifier mismatch.
