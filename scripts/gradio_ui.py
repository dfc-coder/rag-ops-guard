from __future__ import annotations

import time
from pathlib import Path

import gradio as gr

from rag_ops_guard.app import ingestion_service, object_store, query_workflow
from rag_ops_guard.domain.models import QueryContext, QueryRequest
from rag_ops_guard.ingestion.metadata import parse_document


STATUS_LABELS = {
    "answered": "Respondido",
    "insufficient_evidence": "Evidencia insuficiente",
    "clarification_required": "Necesita aclaración",
    "safety_blocked": "Bloqueado por seguridad",
}

TIMING_LABELS = {
    "analysis": "análisis",
    "embedding": "embedding",
    "retrieval": "retrieval",
    "resolver": "resolver",
    "generation": "generación",
}


def _timing_line(timings: dict[str, float]) -> str:
    return " · ".join(
        f"{TIMING_LABELS[key]} {timings[key] / 1000:.1f}s"
        for key in TIMING_LABELS
        if key in timings
    )


def chat(message: str, _history: list, system: str, environment: str) -> str:
    env = environment if environment in {"production", "staging"} else None
    started = time.perf_counter()

    response = query_workflow().invoke(
        QueryRequest(
            question=message,
            context=QueryContext(
                system=system.strip() or None,
                environment=env,
            ),
        )
    )

    elapsed_ms = int(response.timings_ms.get("total", (time.perf_counter() - started) * 1000))
    status = response.status.value
    text = response.answer or response.clarification_question or ""
    citations = "\n".join(f"- **{item.title}** · v{item.version}" for item in response.citations)
    timing_line = _timing_line(response.timings_ms)

    result = f"**{STATUS_LABELS.get(status, status)}** · {elapsed_ms} ms"
    if timing_line:
        result += f"\n\n`{timing_line}`"
    result += f"\n\n{text}"

    if citations:
        result += f"\n\n**Fuentes**\n{citations}"

    return result


def ingest_file(
    filepath: str | None,
    progress=gr.Progress(),  # noqa: B008
) -> str:
    if not filepath:
        return "Seleccioná un archivo Markdown."

    progress(0.2, desc="Leyendo documento")

    path = Path(filepath)
    content = path.read_text(encoding="utf-8")
    metadata, _ = parse_document(content)

    progress(0.5, desc="Guardando documento")

    key = f"raw/uploads/{metadata.id}.md"
    object_store().put_text(key, content, "text/markdown")

    progress(0.7, desc="Generando embeddings")
    result = ingestion_service().ingest(key)

    progress(1.0, desc="Listo")

    return (
        f"**{metadata.title}** ingerido correctamente  \n"
        f"{result.chunks} chunks · versión {metadata.version}"
    )


with gr.Blocks(title="RAG Ops Guard") as demo:
    with gr.Sidebar(open=True):
        gr.Markdown("# RAG Ops Guard")
        gr.Markdown("Local RAG Console")

        system = gr.Textbox(
            label="Sistema",
            placeholder="Opcional: payments, calypso...",
            value="",
        )

        environment = gr.Dropdown(
            choices=[
                ("Cualquiera", ""),
                ("Producción", "production"),
                ("Staging", "staging"),
            ],
            value="",
            label="Ambiente",
        )

        gr.Markdown("### Knowledge base")

        document = gr.File(
            label="Agregar Markdown",
            file_types=[".md", ".markdown"],
            type="filepath",
        )

        ingest_button = gr.Button("Ingerir documento", variant="primary")
        ingest_status = gr.Markdown()

        ingest_button.click(
            ingest_file,
            inputs=document,
            outputs=ingest_status,
            show_progress="full",
        )

    chatbot = gr.Chatbot(
        placeholder=(
            "<strong>RAG Ops Guard</strong><br>Consultá runbooks, APIs, incidentes y SLAs."
        ),
        height=520,
    )

    gr.ChatInterface(
        fn=chat,
        chatbot=chatbot,
        additional_inputs=[system, environment],
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=8000,
        show_error=True,
    )
