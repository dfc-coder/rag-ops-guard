from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import AsyncIterator
from uuid import uuid4

import chainlit as cl
import httpx
from chainlit.input_widget import Select
from langchain_core.messages import HumanMessage, ToolMessage

from rag_ops_guard.agent.react_agent import ReactAgent, ReactStreamEvent
from rag_ops_guard.domain.models import QueryContext

logger = logging.getLogger(__name__)
AGENT = ReactAgent()

_ENVIRONMENTS: dict[str, str | None] = {
    "Cualquier ambiente": None,
    "Producción": "production",
    "Staging": "staging",
}


async def _agent_events(
    message: str,
    *,
    thread_id: str,
    context: QueryContext,
    cancel_event: threading.Event,
) -> AsyncIterator[ReactStreamEvent]:
    """Bridge the synchronous local agent into Chainlit without blocking the event loop."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[object] = asyncio.Queue()
    sentinel = object()

    def worker() -> None:
        stream = AGENT.stream(message, thread_id=thread_id, context=context)
        try:
            for event in stream:
                if cancel_event.is_set():
                    break
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except BaseException as exc:  # noqa: BLE001 - forwarded to the async UI boundary
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
            loop.call_soon_threadsafe(queue.put_nowait, sentinel)

    worker_task = asyncio.create_task(asyncio.to_thread(worker))
    try:
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            if isinstance(item, BaseException):
                raise item
            if isinstance(item, ReactStreamEvent):
                yield item
    finally:
        cancel_event.set()
        if not worker_task.done():
            worker_task.add_done_callback(
                lambda task: task.exception() if not task.cancelled() else None
            )
        else:
            await worker_task


def _query_context() -> QueryContext:
    environment = cl.user_session.get("environment")
    return QueryContext(
        environment=environment if environment in {"production", "staging"} else None
    )


def _last_turn_sources(
    thread_id: str,
    *,
    include_active_evidence: bool = False,
) -> list[dict[str, str]]:
    """Read exact committed evidence; optionally reuse the active EvidenceWindow for transforms."""
    with AGENT._history_guard:  # noqa: SLF001 - UI adapter reads committed agent state
        messages = list(AGENT._histories.get(thread_id, []))  # noqa: SLF001

    last_user = -1
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            last_user = index

    documents: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for message in messages[last_user + 1 :]:
        if not isinstance(message, ToolMessage) or not isinstance(message.content, str):
            continue
        try:
            payload = json.loads(message.content)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or payload.get("supported") is not True:
            continue
        raw_sources = payload.get("sources")
        if not isinstance(raw_sources, list):
            continue

        for raw in raw_sources:
            if not isinstance(raw, dict):
                continue
            title = str(raw.get("title", "Fuente"))
            version = str(raw.get("version", ""))
            system = str(raw.get("system", ""))
            environment = str(raw.get("environment", ""))
            key = (title, version, system, environment)
            section = str(raw.get("section", ""))
            text = str(raw.get("text", ""))
            excerpt = f"### {section}\n{text}" if section else text

            current = documents.get(key)
            if current is None:
                documents[key] = {
                    "title": title,
                    "version": version,
                    "system": system,
                    "environment": environment,
                    "text": excerpt,
                }
            elif excerpt and excerpt not in current["text"]:
                current["text"] += f"\n\n{excerpt}"

    if documents or not include_active_evidence:
        return list(documents.values())

    state = AGENT.grounding_state(thread_id)
    evidence = state.active_evidence()
    if evidence is None:
        return []
    return [
        {
            "title": source.title,
            "version": source.version,
            "system": source.system or "",
            "environment": source.environment or "",
            "text": f"### {source.section}\n{source.text}" if source.section else source.text,
        }
        for source in evidence.sources
    ]


def _source_elements(sources: list[dict[str, str]]) -> tuple[list[cl.Text], str]:
    elements: list[cl.Text] = []
    labels: list[str] = []
    for source in sources:
        title = source["title"]
        meta = " · ".join(
            value
            for value in (
                source["system"],
                source["environment"],
                f"v{source['version']}" if source["version"] else "",
            )
            if value
        )
        content = f"{meta}\n\n{source['text']}".strip()
        elements.append(cl.Text(name=title, content=content, display="side"))
        labels.append(title)
    return elements, " · ".join(labels)


async def _run_turn(message: str) -> None:
    thread_id = cl.user_session.get("thread_id") or str(uuid4())
    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("last_user_message", message)

    cancel_event = threading.Event()
    cl.user_session.set("cancel_event", cancel_event)

    answer = await cl.Message(content="").send()
    streamed_text = ""
    has_visible_output = False

    async with cl.Step(name="Actividad", type="tool", show_input=False) as activity:
        activity.output = "Preparando respuesta…"
        await activity.update()

        try:
            async for event in _agent_events(
                message,
                thread_id=thread_id,
                context=_query_context(),
                cancel_event=cancel_event,
            ):
                if cancel_event.is_set():
                    break

                if event.kind == "status":
                    activity.output = event.text
                    await activity.update()
                    if "base de conocimiento" in event.text.casefold() and has_visible_output:
                        await answer.remove()
                        answer = await cl.Message(content="").send()
                        streamed_text = ""
                        has_visible_output = False
                    continue

                if event.kind == "token":
                    if event.text.startswith(streamed_text):
                        delta = event.text[len(streamed_text) :]
                    else:
                        answer.content = event.text
                        await answer.update()
                        streamed_text = event.text
                        has_visible_output = bool(event.text)
                        continue
                    if delta:
                        await answer.stream_token(delta)
                        streamed_text = event.text
                        has_visible_output = True
                    continue

                if event.kind == "done":
                    sources = _last_turn_sources(
                        thread_id,
                        include_active_evidence=event.policy == "reuse_evidence",
                    )
                    elements, source_labels = _source_elements(sources)
                    final_text = event.text
                    if source_labels:
                        final_text += f"\n\n**Fuentes:** {source_labels}"
                    answer.content = final_text
                    answer.elements = elements
                    await answer.update()
                    activity.output = (
                        f"Listo · {event.elapsed_ms / 1000:.1f}s · "
                        f"{event.tool_calls} tool call{'s' if event.tool_calls != 1 else ''}"
                    )
                    if event.policy == "reuse_evidence":
                        activity.output += " · evidencia reutilizada"
                    if sources:
                        activity.output += (
                            f" · {len(sources)} fuente{'s' if len(sources) != 1 else ''}"
                        )
                    await activity.update()
                    return

                if event.kind == "error":
                    answer.content = event.text
                    await answer.update()
                    activity.output = "Turno recuperable · la conversación anterior se conserva"
                    await activity.update()
                    await cl.Message(
                        content="Podés reintentar este turno sin reiniciar la conversación.",
                        actions=[
                            cl.Action(
                                name="retry_last",
                                payload={"source": "recoverable-turn"},
                                label="Reintentar",
                                icon="rotate-ccw",
                            )
                        ],
                    ).send()
                    return

            if cancel_event.is_set():
                activity.output = "Generación detenida · el turno no se incorporó a la conversación"
                await activity.update()
                if has_visible_output:
                    answer.content = f"{answer.content}\n\n---\n\n_Generación detenida._"
                else:
                    answer.content = "_Generación detenida._"
                await answer.update()
        except asyncio.CancelledError:
            cancel_event.set()
            activity.output = "Generación detenida"
            await activity.update()
            raise
        except Exception:
            logger.exception("Unhandled Chainlit ReAct bridge failure")
            activity.output = "La interfaz perdió el turno, pero la sesión sigue disponible"
            await activity.update()
            notice = "No pude cerrar este turno. La conversación anterior sigue intacta."
            answer.content = (
                f"{answer.content}\n\n---\n\n{notice}" if has_visible_output else notice
            )
            await answer.update()
        finally:
            cl.user_session.set("cancel_event", None)


async def _diagnostics() -> str:
    states: list[tuple[str, str]] = []
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            response = await client.get("http://127.0.0.1:8080/health")
            states.append(
                ("Generación", "Ready" if response.is_success else f"HTTP {response.status_code}")
            )
        except httpx.HTTPError:
            states.append(("Generación", "No disponible"))

        try:
            response = await client.get("http://127.0.0.1:4566/")
            states.append(
                (
                    "Knowledge / Floci",
                    "Ready" if response.status_code < 500 else f"HTTP {response.status_code}",
                )
            )
        except httpx.HTTPError:
            states.append(("Knowledge / Floci", "No disponible"))

        try:
            embedding = await client.post(
                "http://127.0.0.1:8083/v3/embeddings",
                json={
                    "model": "OpenVINO/Qwen3-Embedding-0.6B-int8-ov",
                    "input": "health probe",
                },
            )
            states.append(
                (
                    "Embeddings / OpenVINO",
                    "Ready" if embedding.is_success else f"HTTP {embedding.status_code}",
                )
            )
        except httpx.HTTPError:
            states.append(("Embeddings / OpenVINO", "No disponible"))

        try:
            rerank = await client.post(
                "http://127.0.0.1:8083/v3/rerank",
                json={
                    "model": "OpenVINO/Qwen3-Reranker-0.6B-seq-cls-fp16-ov",
                    "query": "health",
                    "documents": ["health"],
                    "top_n": 1,
                },
            )
            states.append(
                (
                    "Reranker / OpenVINO",
                    "Ready" if rerank.is_success else f"HTTP {rerank.status_code}",
                )
            )
        except httpx.HTTPError:
            states.append(("Reranker / OpenVINO", "No disponible"))

    lines = [f"- **{label}:** {state}" for label, state in states]
    return "### Estado local\n" + "\n".join(lines)


@cl.set_starters
async def starters() -> list[cl.Starter]:
    return [
        cl.Starter(label="Calypso retries", message="¿Cuántos reintentos permite Calypso?"),
        cl.Starter(
            label="Runbooks disponibles",
            message="¿Qué documentación tienes disponible?",
        ),
        cl.Starter(label="Analizar incidente", message="¿Qué sabes del incidente INC-001?"),
        cl.Starter(
            label="Generar código",
            message="Escribe una función corta en Python para merge sort.",
        ),
    ]


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("thread_id", str(uuid4()))
    cl.user_session.set("environment", None)
    cl.user_session.set("cancel_event", None)

    await cl.ChatSettings(
        [
            Select(
                id="environment",
                label="Ambiente",
                values=list(_ENVIRONMENTS),
                initial_index=0,
                description="Filtra la evidencia operacional cuando el agente usa RAG.",
            )
        ]
    ).send()

    await cl.Message(
        content=(
            "## RAG Ops Guard\n"
            "Asistente local para operaciones de integración. Conversá normalmente, pedí código "
            "o consultá runbooks, APIs, incidentes y SLAs."
        ),
        actions=[
            cl.Action(
                name="diagnostics",
                payload={"source": "welcome"},
                label="Estado local",
                icon="activity",
            )
        ],
    ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict[str, object]) -> None:
    label = str(settings.get("environment", "Cualquier ambiente"))
    cl.user_session.set("environment", _ENVIRONMENTS.get(label))


@cl.on_message
async def on_message(message: cl.Message) -> None:
    await _run_turn(message.content)


@cl.on_stop
async def on_stop() -> None:
    cancel_event = cl.user_session.get("cancel_event")
    if isinstance(cancel_event, threading.Event):
        cancel_event.set()


@cl.action_callback("retry_last")
async def retry_last(action: cl.Action) -> None:
    await action.remove()
    message = cl.user_session.get("last_user_message")
    if isinstance(message, str) and message.strip():
        await _run_turn(message)
    else:
        await cl.Message(content="No hay un turno anterior para reintentar.").send()


@cl.action_callback("diagnostics")
async def diagnostics(action: cl.Action) -> None:
    await action.remove()
    await cl.Message(content=await _diagnostics()).send()
