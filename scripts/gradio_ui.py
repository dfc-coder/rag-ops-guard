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
    "route": "route",
    "catalog": "catálogo",
    "search": "search",
    "rewrite": "rewrite",
    "generation": "generación",
}

CSS = """
:root {
    --rag-border: #3a3a3f;
    --rag-panel: #17171a;
    --rag-panel-2: #1f1f23;
    --rag-muted: #a1a1aa;
}

html, body, .gradio-container {
    min-height: 100vh !important;
}

.gradio-container {
    width: 100% !important;
    max-width: none !important;
    margin: 0 !important;
    padding: 0 !important;
}

footer {
    display: none !important;
}

#app-shell {
    width: 100% !important;
    min-height: 100vh;
    padding: 20px 24px 18px 24px;
    box-sizing: border-box;
}

#rag-header {
    width: 100%;
    align-items: center;
    margin-bottom: 14px;
    padding: 0 2px;
}

#rag-header h1 {
    margin: 0;
    font-size: 26px;
    line-height: 1.05;
    letter-spacing: -0.035em;
}

#rag-header p {
    margin: 6px 0 0 0;
    color: var(--rag-muted);
    font-size: 13px;
}

#workspace {
    width: 100%;
    gap: 16px;
    align-items: stretch;
}

#context-panel {
    min-width: 280px !important;
    max-width: 320px !important;
}

#context-panel .block,
#context-panel .form,
#context-panel .gr-group {
    border-color: var(--rag-border) !important;
}

#chat-panel {
    min-width: 0 !important;
    width: 100% !important;
}

#rag-chatbot {
    min-height: calc(100vh - 190px) !important;
    height: calc(100vh - 190px) !important;
    border: 1px solid var(--rag-border) !important;
    border-radius: 8px !important;
    overflow: hidden;
}

#rag-chatbot .message {
    max-width: min(860px, 82%) !important;
}

.rag-observability {
    display: inline-flex;
    align-items: center;
    justify-content: flex-end;
    width: 100%;
    font-size: 12px;
    color: var(--rag-muted);
    white-space: nowrap;
}

.rag-observability strong {
    color: inherit;
    font-weight: 600;
}

@media (max-width: 820px) {
    #app-shell {
        padding: 14px;
    }

    #workspace {
        flex-direction: column !important;
    }

    #context-panel {
        min-width: 100% !important;
        max-width: none !important;
    }

    #rag-chatbot {
        min-height: 58vh !important;
        height: 58vh !important;
    }
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


def _diagnostic_line(response: object) -> str:
    route = getattr(response, "route", None) or "unknown"
    parts = [f"route {route}"]
    confidence = getattr(response, "route_confidence", None)
    if confidence is not None:
        parts.append(f"confidence {confidence:.2f}")
    relevance = getattr(response, "relevance_score", None)
    if route == "knowledge" and relevance is not None:
        parts.append(f"relevance {relevance:.2f}")
    if getattr(response, "rewritten_query", None):
        parts.append("query rewritten")
    return " · ".join(parts)


def chat(
    message: str,
    _history: list,
    system: str,
    environment: str,
    request: gr.Request | None = None,
) -> str:
    env = environment if environment in {"production", "staging"} else None
    started = time.perf_counter()
    thread_id = request.session_hash if request and request.session_hash else None

    response = query_workflow().invoke(
        QueryRequest(
            question=message,
            thread_id=thread_id,
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
    diagnostic_line = _diagnostic_line(response)

    parts: list[str] = []
    status_label = STATUS_LABELS.get(status)
    if status_label:
        parts.append(f"**{status_label}**")

    parts.append(text)

    if response.citations:
        sources = "\n".join(f"- **{item.title}** · v{item.version}" for item in response.citations)
        source_summary = f"Fuentes ({len(response.citations)})"
        parts.append(f"<details><summary>{source_summary}</summary>\n\n{sources}\n\n</details>")

    parts.append(
        f"<details><summary>Detalles</summary>\n\n"
        f"<small>{diagnostic_line} · {timing_line}</small>\n\n</details>"
    )
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
    query_workflow().refresh_knowledge()
    progress(1.0, desc="Listo")

    return f"**{metadata.title}** · {result.chunks} chunks · v{metadata.version}"


def _observability_label() -> str:
    settings = get_settings()
    connected = settings.langsmith_tracing and bool(settings.langsmith_api_key)
    state = "conectado" if connected else "desconectado"
    return f"<div class='rag-observability'>LangSmith · <strong>{state}</strong></div>"


with (
    gr.Blocks(
        title="RAG Ops Guard",
        fill_width=True,
        fill_height=True,
    ) as demo,
    gr.Column(elem_id="app-shell"),
):
    with gr.Row(elem_id="rag-header"):
        with gr.Column(scale=5, min_width=320):
            gr.Markdown(
                "# RAG Ops Guard\n"
                "Agente conversacional para operaciones con respuestas fundamentadas."
            )
        with gr.Column(scale=1, min_width=180):
            gr.HTML(_observability_label())

    with gr.Row(equal_height=True, elem_id="workspace"):
        with gr.Column(scale=1, min_width=280, elem_id="context-panel"):
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

        with gr.Column(scale=5, min_width=500, elem_id="chat-panel"):
            chatbot = gr.Chatbot(
                placeholder=(
                    "<strong>Preguntá o conversá sobre operaciones.</strong><br>"
                    "El agente consulta la knowledge base cuando necesita evidencia."
                ),
                height="calc(100vh - 190px)",
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
        css=CSS,
    )
