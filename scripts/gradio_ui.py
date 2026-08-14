from __future__ import annotations

import time
from pathlib import Path

import gradio as gr

from rag_ops_guard.app import ingestion_service, object_store, query_workflow
from rag_ops_guard.config import get_settings
from rag_ops_guard.domain.models import QueryContext, QueryRequest
from rag_ops_guard.ingestion.metadata import parse_document


STATUS_LABELS = {
    "insufficient_evidence": "Evidencia insuficiente",
    "clarification_required": "Necesita aclaración",
    "safety_blocked": "Bloqueado por seguridad",
}

TIMING_LABELS = {
    "embedding": "embedding",
    "retrieval": "retrieval",
    "resolver": "resolver",
    "generation": "generación",
}

CSS = """
.gradio-container {
    max-width: 1240px !important;
    margin: 0 auto !important;
}
footer { display: none !important; }
#rag-header {
    padding: 18px 4px 10px 4px;
}
#rag-header h1 {
    margin-bottom: 2px;
    font-size: 24px;
    letter-spacing: -0.02em;
}
#rag-header p {
    margin: 0;
    opacity: 0.66;
}
#context-panel {
    min-width: 245px;
    max-width: 280px;
}
#context-panel > div {
    border-radius: 14px;
}
#chat-panel {
    min-width: 0;
}
#rag-chatbot {
    border-radius: 16px !important;
    overflow: hidden;
}
.rag-observability {
    font-size: 12px;
    opacity: 0.72;
    text-align: right;
    padding-top: 8px;
}
"""


def _timing_line(timings: dict[str, float], elapsed_ms: int) -> str:
    stages = " · ".join(
        f"{TIMING_LABELS[key]} {timings[key] / 1000:.1f}s"
        for key in TIMING_LABELS
        if key in timings
    )
    total = f"{elapsed_ms / 1000:.1f}s total"
    return f"{total} · {stages}" if stages else total


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
    timing_line = _timing_line(response.timings_ms, elapsed_ms)

    parts: list[str] = []
    status_label = STATUS_LABELS.get(status)
    if status_label:
        parts.append(f"**{status_label}**")

    parts.append(text)

    if response.citations:
        sources = "\n".join(
            f"- **{item.title}** · v{item.version}" for item in response.citations
        )
        parts.append(
            f"<details><summary>Fuentes ({len(response.citations)})</summary>\n\n{sources}\n\n</details>"
        )

    parts.append(f"<small>{timing_line}</small>")
    return "\n\n".join(part for part in parts if part)


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

    return f"**{metadata.title}** · {result.chunks} chunks · v{metadata.version}"


def _observability_label() -> str:
    settings = get_settings()
    connected = settings.langsmith_tracing and bool(settings.langsmith_api_key)
    state = "conectado" if connected else "desconectado"
    return f"<div class='rag-observability'>LangSmith · {state}</div>"


with gr.Blocks(title="RAG Ops Guard", css=CSS) as demo:
    with gr.Row(elem_id="rag-header"):
        with gr.Column(scale=4):
            gr.Markdown("# RAG Ops Guard\nConsultá la knowledge base operativa con respuestas fundamentadas.")
        with gr.Column(scale=1):
            gr.HTML(_observability_label())

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=245, elem_id="context-panel"):
            with gr.Group():
                gr.Markdown("### Contexto")
                system = gr.Textbox(
                    label="Sistema",
                    placeholder="Opcional, ej. payments",
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

            with gr.Accordion("Knowledge base", open=False):
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

        with gr.Column(scale=4, elem_id="chat-panel"):
            chatbot = gr.Chatbot(
                placeholder=(
                    "<strong>Preguntá sobre runbooks, APIs, incidentes o SLAs.</strong><br>"
                    "Las respuestas se limitan a la evidencia disponible."
                ),
                height=570,
                show_label=False,
                elem_id="rag-chatbot",
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
