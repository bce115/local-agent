"""
Web server for the local agent — streaming, persistence, memory, model switching.

Streaming: reply is read token-by-token and pushed to the browser as generated.
Model switching: the current model lives in STATE and can be changed at runtime
from the UI dropdown. The conversation is model-agnostic, so switching just
means the next turn uses a different brain — the tools, memory, and gate are
unchanged.

Persistence:
  history.json  raw transcript, saved every turn, reloaded on startup.
  memory.md     curated facts written by the `remember` tool, injected into the
                system prompt every session.

    pip install openai fastapi "uvicorn[standard]"
    ollama pull devstral-small-2
    python server.py            # then open http://localhost:8000
"""

from __future__ import annotations

import asyncio
import json
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI

import tools

BASE_URL = "http://localhost:11434/v1"
TEMPERATURE = 0.7
MAX_TOOL_HOPS = 8
HISTORY_FILE = "history.json"

# The active model. Mutable so the UI can switch it at runtime. The rest of the
# code reads STATE["model"] instead of a constant.
STATE = {"model": "devstral-small-2"}

BASE_SYSTEM_PROMPT = """You are a local assistant running on the user's own machine.
You have tools to run shell commands, read and search local files, fetch web
pages, and remember durable facts. Use tools when they help; answer directly
when they don't. Be concise and technical. When something about the user or
their setup is worth recalling later (a preference, a name, a path, an ongoing
project), call the remember tool. When you run a command, briefly say why."""

client = AsyncOpenAI(base_url=BASE_URL, api_key="ollama")
app = FastAPI()


# --- models ---------------------------------------------------------------

async def list_models() -> list:
    """Ask Ollama which models are installed (via the OpenAI-compatible endpoint)."""
    try:
        resp = await client.models.list()
        names = sorted(m.id for m in resp.data)
        if STATE["model"] not in names:
            names.insert(0, STATE["model"])
        return names
    except Exception:  # noqa: BLE001
        return [STATE["model"]]


# --- memory + history -----------------------------------------------------

def build_system_prompt() -> str:
    mem = ""
    if os.path.exists(tools.MEMORY_FILE):
        with open(tools.MEMORY_FILE, encoding="utf-8") as f:
            mem = f.read().strip()
    if mem:
        return BASE_SYSTEM_PROMPT + "\n\nThings you remember from past sessions:\n" + mem
    return BASE_SYSTEM_PROMPT


def load_history() -> list:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, encoding="utf-8") as f:
                msgs = json.load(f)
            if msgs and msgs[0].get("role") == "system":
                msgs[0]["content"] = build_system_prompt()
            else:
                msgs.insert(0, {"role": "system", "content": build_system_prompt()})
            return msgs
        except Exception:  # noqa: BLE001
            pass
    return [{"role": "system", "content": build_system_prompt()}]


def save_history(messages: list) -> None:
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2)
    except Exception:  # noqa: BLE001
        pass


def replay(messages: list) -> list:
    return [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]


# --- routes ---------------------------------------------------------------

@app.get("/")
async def index() -> FileResponse:
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.websocket("/ws")
async def ws(sock: WebSocket) -> None:
    await sock.accept()
    messages = load_history()
    await sock.send_json({"type": "history", "turns": replay(messages)})
    await sock.send_json({"type": "models", "available": await list_models(), "current": STATE["model"]})
    try:
        while True:
            incoming = await sock.receive_json()
            kind = incoming.get("type")
            if kind == "reset":
                messages = [{"role": "system", "content": build_system_prompt()}]
                save_history(messages)
                await sock.send_json({"type": "cleared"})
            elif kind == "set_model":
                STATE["model"] = incoming["model"]
                await sock.send_json({"type": "model_set", "model": STATE["model"]})
            elif kind == "user":
                messages.append({"role": "user", "content": incoming["content"]})
                await run_turn(sock, messages)
                save_history(messages)
    except WebSocketDisconnect:
        return


async def stream_completion(sock: WebSocket, messages: list) -> tuple[str, list]:
    """One streamed model call. Sends tokens as they arrive; returns the full
    text and any tool calls (reassembled from fragments)."""
    stream = await client.chat.completions.create(
        model=STATE["model"], messages=messages, tools=tools.TOOLS,
        temperature=TEMPERATURE, stream=True,
    )
    content = ""
    tcs: dict[int, dict] = {}
    started = False
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            if not started:
                await sock.send_json({"type": "assistant_start"})
                started = True
            content += delta.content
            await sock.send_json({"type": "token", "content": delta.content})
        if delta.tool_calls:
            for tc in delta.tool_calls:
                slot = tcs.setdefault(tc.index, {"id": None, "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        slot["name"] += tc.function.name
                    if tc.function.arguments:
                        slot["arguments"] += tc.function.arguments
    if started:
        await sock.send_json({"type": "assistant_end"})
    ordered = [tcs[i] for i in sorted(tcs)]
    return content, ordered


async def run_turn(sock: WebSocket, messages: list) -> None:
    for _ in range(MAX_TOOL_HOPS):
        await sock.send_json({"type": "status", "state": "thinking"})
        content, tcs = await stream_completion(sock, messages)

        assistant_msg: dict = {"role": "assistant", "content": content or None}
        if tcs:
            assistant_msg["tool_calls"] = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in tcs
            ]
        messages.append(assistant_msg)

        if not tcs:
            await sock.send_json({"type": "status", "state": "idle"})
            return

        for tc in tcs:
            name = tc["name"]
            try:
                args = json.loads(tc["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}

            approved = True
            if name in tools.REQUIRES_APPROVAL:
                await sock.send_json({"type": "tool_request", "id": tc["id"], "name": name, "args": args})
                decision = await sock.receive_json()
                approved = bool(decision.get("approved")) and decision.get("id") == tc["id"]
            else:
                await sock.send_json({"type": "tool_run", "name": name, "args": args})

            if not approved:
                result = "DENIED: user rejected this call."
            else:
                fn = tools.TOOL_FNS.get(name)
                result = await asyncio.to_thread(fn, **args) if fn else f"ERROR: unknown tool {name!r}"

            await sock.send_json({"type": "tool_result", "name": name, "content": str(result)[:2000]})
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": str(result)})

    await sock.send_json({"type": "assistant", "content": "(hit tool-hop limit; stopping this turn)"})
    await sock.send_json({"type": "status", "state": "idle"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
