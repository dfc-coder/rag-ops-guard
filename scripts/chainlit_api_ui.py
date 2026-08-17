from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import chainlit as cl
import httpx
from chainlit.input_widget import Select

ROOT = Path(__file__).resolve().parents[1]
API_FILE = ROOT / ".local" / "api-url"
QUERY_TIMEOUT_SECONDS = float(os.environ.get("UI_QUERY_TIMEOUT_SECONDS", "300"))

_ENVIRONMENTS: dict[str, str | None] = {
    "Cualquier ambiente": None,
    "Producción": "production",
    "Staging": "staging",
}


def _api_url() -> str:
    configured = os.environ.get("RAG_API_URL", "").strip()
    if configured:
        return configured.rstrip("/")
    if API_FILE.is_file():
        return API_FILE.read_text(encoding="utf-8").strip().rstrip("/")
    raise RuntimeError("API local no provisionada. Ejecutá `make up` primero.")


def _context() -> dict[str, str]:
    environment = cl.user_session.get("environment")
    context: dict[str, str] = {}
    if environment in {"production", "staging"}:
        context["environment"] = environment
    return context


def _citation_elements(citations: list[object]) -> tuple[list[cl.Text], str]:
    elements: list[cl.Text] = []
    labels: list[str] = []
    for index, raw in enumerate(citations, start=1):
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title") or raw.get("logical_id") or f"Fuente {index}")
        version = str(raw.get("version") or "")
        chunk_id = str(raw.get("chunk_id") or "")
        logical_id = str(raw.get("logical_id") or "")
        s3_key = str(raw.get("s3_key") or "")
        details = "\n".join(
            line
            for line in (
                f"**Documento:** {title}",
                f"**Logical ID:** {logical_id}" if logical_id else "",
                f"**Versión:** {version}" if version else "",
                f"**Chunk:** `{chunk_id}`" if chunk_id else "",
                f"**S3:** `{s3_key}`" if s3_key else "",
            )
            if line
        )
        elements.append(cl.Text(name=title, content=details, display="side"))
        labels.append(title)
    return elements, " · ".join(dict.fromkeys(labels))


async def _run_turn(message: str) -> None:
    thread_id = cl.user_session.get("thread_id") or str(uuid4())
    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("last_user_message", message)

    async with cl.Step(name="API /v1/query", type="tool", show_input=False) as activity:
        activity.output = "Floci → Lambda → Agent…"
        await activity.update()

        try:
            url = f"{_api_url()}/v1/query"
            payload = {
                "question": message,
                "context": _context(),
                "thread_id": thread_id,
            }
            async with httpx.AsyncClient(timeout=QUERY_TIMEOUT_SECONDS) as client:
                response = await client.post(url, json=payload)

            if response.status_code >= 400:
                detail = response.text.strip() or "<empty body>"
                activity.output = f"HTTP {response.status_code} desde el data plane"
                await activity.update()
                await cl.Message(
                    content=f"**Error HTTP {response.status_code}**\n\n```text\n{detail[:4000]}\n```"
                ).send()
                return

            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("/v1/query devolvió un payload JSON inesperado")

            status = str(data.get("status") or "unknown")
            route = str(data.get("route") or "unknown")
            answer = data.get("answer")
            text = str(answer) if answer else str(data.get("message") or "Sin respuesta")
            citations = data.get("citations")
            citation_list = citations if isinstance(citations, list) else []
            elements, source_labels = _citation_elements(citation_list)

            if source_labels:
                text += f"\n\n**Fuentes:** {source_labels}"

            await cl.Message(content=text, elements=elements).send()

            timings = data.get("timings_ms")
            total_ms = timings.get("total") if isinstance(timings, dict) else None
            timing = f" · {float(total_ms) / 1000:.1f}s" if isinstance(total_ms, (int, float)) else ""
            activity.output = (
                f"{status} · route={route}{timing} · "
                f"{len(citation_list)} cita{'s' if len(citation_list) != 1 else ''}"
            )
            await activity.update()
        except httpx.TimeoutException:
            activity.output = f"Timeout de UI después de {QUERY_TIMEOUT_SECONDS:.0f}s"
            await activity.update()
            await cl.Message(
                content=(
                    "La UI dejó de esperar la respuesta. Revisá `make logs`; "
                    "si Floci informa un timeout de Lambda, el problema está en el data plane/runtime."
                )
            ).send()
        except httpx.HTTPError as exc:
            activity.output = "No se pudo conectar con el data plane"
            await activity.update()
            await cl.Message(content=f"Error de conexión: `{exc}`\n\nEjecutá `make status`.").send()
        except Exception as exc:
            activity.output = "Respuesta inválida desde el data plane"
            await activity.update()
            await cl.Message(content=f"Error: `{type(exc).__name__}: {exc}`").send()


@cl.set_starters
async def starters() -> list[cl.Starter]:
    return [
        cl.Starter(label="Calypso retries", message="¿Cuántos reintentos permite Calypso?"),
        cl.Starter(label="Documentos", message="¿Qué documentación tienes disponible?"),
        cl.Starter(label="Código directo", message="Escribe una función corta en Python para merge sort."),
    ]


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
                description="Se envía como QueryContext al mismo /v1/query usado por Golden.",
            )
        ]
    ).send()

    try:
        api = _api_url()
        api_line = f"`{api}/v1/query`"
    except RuntimeError:
        api_line = "**NO PROVISIONADA** — ejecutá `make up`"

    await cl.Message(
        content=(
            "## RAG Ops Guard · canonical data plane\n\n"
            "Esta UI no ejecuta el agente directamente. Cada mensaje recorre "
            "**Chainlit → Floci API Gateway → Lambda → Agent → Qwen/Retrieval**.\n\n"
            f"API actual: {api_line}"
        ),
        actions=[
            cl.Action(
                name="diagnostics",
                payload={"source": "welcome"},
                label="Estado",
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


@cl.action_callback("retry_last")
async def retry_last(action: cl.Action) -> None:
    await action.remove()
    message = cl.user_session.get("last_user_message")
    if isinstance(message, str) and message.strip():
        await _run_turn(message)


@cl.action_callback("diagnostics")
async def diagnostics(action: cl.Action) -> None:
    await action.remove()
    try:
        api = _api_url()
        message = (
            "### Data plane\n"
            f"- **API:** `{api}`\n"
            "- **Ruta UI:** Chainlit → `/v1/query` → Floci → Lambda → Agent\n"
            "- **Golden:** usa la misma `/v1/query`\n\n"
            "Para runtime/containers/modelo real: ejecutá `make status`."
        )
    except RuntimeError as exc:
        message = f"### Data plane\n- **Estado:** no disponible\n- **Detalle:** {exc}"
    await cl.Message(content=message).send()
