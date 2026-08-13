from __future__ import annotations

import json
from pathlib import Path

from rag_ops_guard.app import (
    chat_model,
    embeddings,
    ingestion_service,
    object_store,
    query_workflow,
    vector_store,
)
from rag_ops_guard.domain.models import Evidence, QueryContext, QueryRequest
from rag_ops_guard.graph.prompts import analysis_prompt, answer_prompt
from rag_ops_guard.retrieval.query_instruction import embedding_query
from rag_ops_guard.retrieval.resolver import EvidenceResolver

QUESTION = "¿Puedo reintentar un pago Orion que ya está SETTLED?"
CONTEXT = QueryContext(system="payments", environment="production")
FIXTURE = Path("evaluation/fixtures/orion-payment-retry.md")


def serialize_evidence(items: list[Evidence]) -> list[dict[str, object]]:
    return [
        {
            "document_id": item.chunk.metadata.id,
            "logical_id": item.chunk.logical_id,
            "chunk_id": item.chunk.id,
            "distance": item.distance,
            "text": item.chunk.text,
        }
        for item in items
    ]


def main() -> None:
    content = FIXTURE.read_text(encoding="utf-8")
    key = "raw/evaluation/orion-payment-retry.md"
    object_store().put_text(key, content, "text/markdown")
    ingest = ingestion_service().ingest(key)

    analysis = chat_model().analyze_query(analysis_prompt(QUESTION, CONTEXT))
    vector = embeddings().embed_query(embedding_query(analysis.normalized_question))
    retrieved = vector_store().query(vector, top_k=8)
    resolved = EvidenceResolver().resolve(retrieved, CONTEXT, limit=5)
    generated = (
        chat_model().generate_answer(answer_prompt(QUESTION, resolved)) if resolved else None
    )
    final = query_workflow().invoke(QueryRequest(question=QUESTION, context=CONTEXT))

    trace = {
        "question": QUESTION,
        "ingest": ingest.model_dump(mode="json"),
        "analysis": analysis.model_dump(mode="json"),
        "retrieval_hit": any(item.chunk.logical_id == "orion-payment-retry" for item in retrieved),
        "retrieved": serialize_evidence(retrieved),
        "resolution_hit": any(item.chunk.logical_id == "orion-payment-retry" for item in resolved),
        "resolved": serialize_evidence(resolved),
        "generation": generated.model_dump(mode="json") if generated else None,
        "final": final.model_dump(mode="json"),
        "source_hit": any(c.logical_id == "orion-payment-retry" for c in final.citations),
    }

    output = Path("artifacts/accuracy")
    output.mkdir(parents=True, exist_ok=True)
    (output / "orion-trace.json").write_text(
        json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(trace, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
