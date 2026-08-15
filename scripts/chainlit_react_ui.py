from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from uuid import uuid4

import chainlit as cl
from chainlit.input_widget import Select

from rag_ops_guard.agent.react_agent import ReactAgent, ReactStreamEvent
from rag_ops_guard.domain.models import QueryContext

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
) -> AsyncIterator[ReactStreamEvent]:
    """Bridge the synchronous local-agent stream into Chainlit without blocking its event loop."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[object] = asyncio.Queue()
    sentinel = object()

    def worker() -> None:
        try:
            for event in AGENT.stream(message, thread_id=thread_id, context=context):
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except BaseException as exc:  # noqa: BLE001 - propagated to the async UI boundary
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
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
        await worker_task


def _query_context() -> QueryContext:
    environment = cl.user_session.get("environment")
    return QueryContext(environment=environment if environment in {"production", "staging"} else None)


async def _run_turn(message: str) -> None:
    thread_id = cl.user_session.get("thread_id") or str(uuid4())
    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("last_user_message", message)

    answer = await cl.Message(content="").send()
    streamed_text = ""
    has_visible_output = False

    async with cl.Step(name="RAG Ops Guard", type="run") as activity:
        activity.output = "Preparando respuesta…"
        await activity.update()

        try:
            async for event in _agent_events(
                message,
                thread_id=thread_id,
                context=_query_context(),
            ):
                if event.kind == "status":
                    activity.output = event.text
                    await activity.update()

                    # A tool decision makes any pre-tool assistant text provisional. Replace the
                    # message before streaming the grounded answer so the user never sees both.
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
                        # Defensive reset if the agent intentionally replaced provisional text.
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
                    answer.content = event.text
                    await answer.update()
                    activity.output = (
                        f"Listo · {event.elapsed_ms / 1000:.1f}s · "
                        f"{event.tool_calls} tool call{'s' if event.tool_calls != 1 else ''}"
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
        except Exception:
            activity.output = "La interfaz perdió el turno, pero la sesión sigue disponible"
            await activity.update()
            if has_visible_output:
                answer.content = (
                    f"{answer.content}\n\n---\n\n"
                    "No pude cerrar este turno. La conversación anterior sigue intacta."
                )
            else:
                answer.content = (
                    "No pude completar este turno. La conversación anterior sigue intacta."
                )
            await answer.update()


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("thread_id", str(uuid4()))
    cl.user_session.set("environment", None)

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
            "Asistente local para operaciones de integración. Podés conversar normalmente, "
            "pedir código o consultar runbooks, APIs, incidentes y SLAs.\n\n"
            "**Probá:** `¿Cuántos reintentos permite Calypso?` o "
            "`Escribe una función Perl para Caesar cipher.`"
        )
    ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict[str, object]) -> None:
    label = str(settings.get("environment", "Cualquier ambiente"))
    cl.user_session.set("environment", _ENVIRONMENTS.get(label))


@cl.on_message
async def on_message(message: cl.Message) -> None:
    await _run_turn(message.content)


@cl.action_callback("retry_last")
async def retry_last(action: cl.Action) -> None:
    await action.remove()
    message = cl.user_session.get("last_user_message")
    if isinstance(message, str) and message.strip():
        await _run_turn(message)
    else:
        await cl.Message(content="No hay un turno anterior para reintentar.").send()
