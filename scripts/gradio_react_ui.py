from __future__ import annotations

import logging
from collections.abc import Iterator
from uuid import uuid4

import gradio as gr

from rag_ops_guard.agent.react_agent import ReactAgent
from rag_ops_guard.domain.models import QueryContext

logger = logging.getLogger(__name__)
AGENT = ReactAgent()

CSS = """
html, body, .gradio-container { min-height: 100vh !important; }
.gradio-container { max-width: none !important; margin: 0 !important; }
footer { display: none !important; }
#shell { padding: 18px 24px; }
#chat { min-height: calc(100vh - 210px) !important; height: calc(100vh - 210px) !important; }
#side { min-width: 260px !important; max-width: 300px !important; }
@media (max-width: 820px) {
  #shell { padding: 12px; }
  #workspace { flex-direction: column !important; }
  #side { min-width: 100% !important; max-width: none !important; }
  #chat { min-height: 60vh !important; height: 60vh !important; }
}
"""


def _render_terminal(text: str, *, tool_calls: int, elapsed_ms: int, failed: bool) -> str:
    state = "turno recuperable · memoria anterior conservada" if failed else "streaming completo"
    return (
        f"{text}\n\n"
        f"<details><summary>Detalles</summary>\n\n"
        f"<small>ReAct · tools {tool_calls} · {elapsed_ms / 1000:.1f}s · {state}</small>"
        f"\n\n</details>"
    )


def chat(message: str, _history: list, environment: str, thread_id: str) -> Iterator[str]:
    """Yield replacement responses so Gradio paints progress and model tokens immediately."""
    del _history
    env = environment if environment in {"production", "staging"} else None
    try:
        for event in AGENT.stream(
            message,
            thread_id=thread_id,
            context=QueryContext(environment=env),
        ):
            if event.kind == "status":
                yield f"_{event.text}_"
            elif event.kind == "token":
                yield event.text
            elif event.kind == "done":
                yield _render_terminal(
                    event.text,
                    tool_calls=event.tool_calls,
                    elapsed_ms=event.elapsed_ms,
                    failed=False,
                )
            elif event.kind == "error":
                yield _render_terminal(
                    event.text,
                    tool_calls=event.tool_calls,
                    elapsed_ms=event.elapsed_ms,
                    failed=True,
                )
    except Exception:
        # Last UI boundary: backend exceptions should never become Gradio's generic red Error.
        logger.exception("Unhandled ReAct UI failure; keeping the current conversation usable")
        yield (
            "No pude completar este turno por un problema de la interfaz. "
            "La conversación anterior se conservó; podés volver a intentarlo."
        )


def clear(thread_id: str) -> str:
    AGENT.clear_thread(thread_id)
    return str(uuid4())


with (
    gr.Blocks(title="RAG Ops Guard · ReAct Beta", fill_width=True, fill_height=True) as demo,
    gr.Column(elem_id="shell"),
):
    thread_id = gr.State(lambda: str(uuid4()))
    gr.Markdown(
        "# RAG Ops Guard · ReAct Beta\n"
        "Agente conversacional con memoria de hilo y RAG como herramienta."
    )

    with gr.Row(elem_id="workspace"):
        with gr.Column(scale=1, min_width=260, elem_id="side"):
            environment = gr.Dropdown(
                choices=[
                    ("Cualquier ambiente", ""),
                    ("Producción", "production"),
                    ("Staging", "staging"),
                ],
                value="",
                label="Contexto",
            )
            gr.Markdown(
                "**Runtime**\n\n"
                "- LLM: Qwen 3.5 2B / CPU\n"
                "- Embeddings: OpenVINO / Intel GPU\n"
                "- Reranker: OpenVINO / Intel GPU\n"
                "- RAG: `search_knowledge` tool"
            )

        with gr.Column(scale=5, min_width=500):
            chatbot = gr.Chatbot(
                placeholder=(
                    "<strong>Conversá normalmente.</strong><br>"
                    "El agente decide cuándo consultar la knowledge base."
                ),
                height="calc(100vh - 210px)",
                show_label=False,
                elem_id="chat",
            )
            gr.ChatInterface(
                fn=chat,
                chatbot=chatbot,
                additional_inputs=[environment, thread_id],
            )
            chatbot.clear(clear, inputs=[thread_id], outputs=[thread_id], queue=False)


if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=8000,
        show_error=False,
        css=CSS,
    )
