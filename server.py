"""
Web server for the local agent — streaming, persistence, memory, model switching.

Streaming: reply is read token-by-token and pushed to the browser as generated.
Model switching: the current model lives in STATE and can be changed at runtime
from the UI dropdown. The conversation is model-agnostic, so switching just
means the next turn uses a different brain — the tools, memory, and gate are
unchanged.

Persistence:
  chats/<id>.json   one transcript per named chat, saved every turn.
  chats/index.json  chat list (id, title, updated_at).
  history.json      legacy single transcript; imported once into chats/ if present.
  memory.md         curated facts written by the `remember` tool (or the Memory
                    panel), injected into the system prompt every session.

    pip install openai fastapi "uvicorn[standard]"
    ollama pull devstral-small-2
    python server.py            # then open http://localhost:8000
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI

import tools

BASE_URL = "http://localhost:11434/v1"
TEMPERATURE = 0.7
MAX_TOOL_HOPS = 8
LEGACY_HISTORY = "history.json"
CHATS_DIR = "chats"
INDEX_FILE = os.path.join(CHATS_DIR, "index.json")

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


@dataclass
class Conn:
    sock: WebSocket
    inbox: asyncio.Queue = field(default_factory=asyncio.Queue)
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    busy: bool = False
    chat_id: str = ""
    messages: list = field(default_factory=list)


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


# --- memory ---------------------------------------------------------------

def build_system_prompt() -> str:
    mem = read_memory().strip()
    if mem:
        return BASE_SYSTEM_PROMPT + "\n\nThings you remember from past sessions:\n" + mem
    return BASE_SYSTEM_PROMPT


def read_memory() -> str:
    if os.path.exists(tools.MEMORY_FILE):
        with open(tools.MEMORY_FILE, encoding="utf-8") as f:
            return f.read()
    return ""


def write_memory(content: str) -> None:
    with open(tools.MEMORY_FILE, "w", encoding="utf-8") as f:
        f.write(content)


def clear_memory() -> None:
    if os.path.exists(tools.MEMORY_FILE):
        os.remove(tools.MEMORY_FILE)


def refresh_system_prompt(messages: list) -> None:
    prompt = build_system_prompt()
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = prompt
    else:
        messages.insert(0, {"role": "system", "content": prompt})


# --- chats ----------------------------------------------------------------

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_id(cid: str) -> str:
    return "".join(c for c in cid if c.isalnum())[:32]


def title_from_messages(messages: list) -> str:
    for m in messages:
        if m.get("role") == "user" and m.get("content"):
            line = str(m["content"]).strip().splitlines()[0]
            return (line[:60] + "…") if len(line) > 60 else line or "New chat"
    return "New chat"


def empty_messages() -> list:
    return [{"role": "system", "content": build_system_prompt()}]


def load_index() -> list:
    os.makedirs(CHATS_DIR, exist_ok=True)
    migrate_legacy()
    if os.path.exists(INDEX_FILE):
        try:
            with open(INDEX_FILE, encoding="utf-8") as f:
                items = json.load(f)
            if isinstance(items, list):
                return items
        except Exception:  # noqa: BLE001
            pass
    return []


def save_index(items: list) -> None:
    os.makedirs(CHATS_DIR, exist_ok=True)
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)


def chat_path(cid: str) -> str:
    return os.path.join(CHATS_DIR, f"{_safe_id(cid)}.json")


def load_chat(cid: str) -> list:
    path = chat_path(cid)
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                msgs = json.load(f)
            if not isinstance(msgs, list):
                msgs = []
            refresh_system_prompt(msgs)
            for m in msgs:
                if m.get("role") == "assistant" and m.get("content") is None:
                    m["content"] = ""
            return msgs
        except Exception:  # noqa: BLE001
            pass
    return empty_messages()


def save_chat(cid: str, messages: list, items: list, *, auto_title: bool = True) -> list:
    os.makedirs(CHATS_DIR, exist_ok=True)
    with open(chat_path(cid), "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2)
    found = None
    for it in items:
        if it.get("id") == cid:
            found = it
            break
    if found is None:
        found = {"id": cid, "title": "New chat", "updated_at": _now(), "auto": True}
        items.insert(0, found)
    found["updated_at"] = _now()
    if auto_title and found.get("auto", True):
        derived = title_from_messages(messages)
        if derived != "New chat":
            found["title"] = derived
    items[:] = [found] + [it for it in items if it.get("id") != cid]
    save_index(items)
    return items


def delete_chat_file(cid: str) -> None:
    path = chat_path(cid)
    if os.path.exists(path):
        os.remove(path)


def migrate_legacy() -> None:
    if os.path.exists(INDEX_FILE):
        return
    os.makedirs(CHATS_DIR, exist_ok=True)
    others = [
        n for n in os.listdir(CHATS_DIR)
        if n.endswith(".json") and n != "index.json"
    ]
    if others:
        return
    if not os.path.exists(LEGACY_HISTORY):
        return
    try:
        with open(LEGACY_HISTORY, encoding="utf-8") as f:
            msgs = json.load(f)
        if not isinstance(msgs, list):
            return
    except Exception:  # noqa: BLE001
        return
    cid = uuid.uuid4().hex[:12]
    refresh_system_prompt(msgs)
    with open(chat_path(cid), "w", encoding="utf-8") as f:
        json.dump(msgs, f, indent=2)
    save_index([{
        "id": cid,
        "title": title_from_messages(msgs),
        "updated_at": _now(),
        "auto": True,
    }])


def new_chat(items: list) -> tuple[str, list, list]:
    cid = uuid.uuid4().hex[:12]
    messages = empty_messages()
    items = save_chat(cid, messages, items)
    return cid, messages, items


def public_chats(items: list) -> list:
    return [
        {"id": it["id"], "title": it.get("title") or "New chat", "updated_at": it.get("updated_at", "")}
        for it in items
    ]


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


async def send(conn: Conn, payload: dict) -> None:
    try:
        await conn.sock.send_json(payload)
    except Exception:  # noqa: BLE001
        pass


async def emit_chats(conn: Conn, items: list) -> None:
    await send(conn, {"type": "chats", "items": public_chats(items), "current": conn.chat_id})


async def emit_history(conn: Conn) -> None:
    await send(conn, {"type": "history", "turns": replay(conn.messages)})


async def reader(conn: Conn) -> None:
    try:
        while True:
            incoming = await conn.sock.receive_json()
            kind = incoming.get("type")
            if kind == "cancel":
                conn.cancel.set()
                await conn.inbox.put(incoming)
                continue
            if "approved" in incoming and incoming.get("id"):
                await conn.inbox.put(incoming)
                continue
            if conn.busy:
                continue
            await conn.inbox.put(incoming)
    except WebSocketDisconnect:
        conn.cancel.set()
        await conn.inbox.put(None)
    except Exception:  # noqa: BLE001
        conn.cancel.set()
        await conn.inbox.put(None)


@app.websocket("/ws")
async def ws(sock: WebSocket) -> None:
    await sock.accept()
    conn = Conn(sock=sock)
    items = load_index()
    if not items:
        conn.chat_id, conn.messages, items = new_chat(items)
    else:
        conn.chat_id = items[0]["id"]
        conn.messages = load_chat(conn.chat_id)
    await emit_chats(conn, items)
    await emit_history(conn)
    await send(conn, {"type": "models", "available": await list_models(), "current": STATE["model"]})

    task = asyncio.create_task(reader(conn))
    try:
        while True:
            incoming = await conn.inbox.get()
            if incoming is None:
                return
            kind = incoming.get("type")
            if kind == "cancel":
                continue
            if kind == "user":
                text = (incoming.get("content") or "").strip()
                if not text:
                    continue
                conn.busy = True
                conn.cancel.clear()
                conn.messages.append({"role": "user", "content": text})
                await run_turn(conn)
                items = save_chat(conn.chat_id, conn.messages, items)
                await emit_chats(conn, items)
                conn.busy = False
            elif kind == "set_model":
                STATE["model"] = incoming["model"]
                await send(conn, {"type": "model_set", "model": STATE["model"]})
            elif kind == "new_chat":
                conn.chat_id, conn.messages, items = new_chat(items)
                await emit_chats(conn, items)
                await emit_history(conn)
            elif kind == "switch_chat":
                cid = _safe_id(incoming.get("id") or "")
                if not cid or not any(it.get("id") == cid for it in items):
                    continue
                conn.chat_id = cid
                conn.messages = load_chat(cid)
                await emit_chats(conn, items)
                await emit_history(conn)
            elif kind == "rename_chat":
                cid = _safe_id(incoming.get("id") or "")
                title = (incoming.get("title") or "").strip()[:80] or "New chat"
                for it in items:
                    if it.get("id") == cid:
                        it["title"] = title
                        it["auto"] = False
                        it["updated_at"] = _now()
                        break
                save_index(items)
                await emit_chats(conn, items)
            elif kind == "delete_chat":
                cid = _safe_id(incoming.get("id") or "")
                items = [it for it in items if it.get("id") != cid]
                delete_chat_file(cid)
                if not items:
                    conn.chat_id, conn.messages, items = new_chat(items)
                else:
                    save_index(items)
                    if conn.chat_id == cid:
                        conn.chat_id = items[0]["id"]
                        conn.messages = load_chat(conn.chat_id)
                await emit_chats(conn, items)
                await emit_history(conn)
            elif kind == "get_memory":
                await send(conn, {"type": "memory", "content": read_memory()})
            elif kind == "set_memory":
                write_memory(incoming.get("content") or "")
                refresh_system_prompt(conn.messages)
                items = save_chat(conn.chat_id, conn.messages, items, auto_title=False)
                await send(conn, {"type": "memory", "content": read_memory()})
            elif kind == "clear_memory":
                clear_memory()
                refresh_system_prompt(conn.messages)
                items = save_chat(conn.chat_id, conn.messages, items, auto_title=False)
                await send(conn, {"type": "memory", "content": ""})
            elif kind == "export":
                await send(conn, {
                    "type": "export_data",
                    "id": conn.chat_id,
                    "title": next((it.get("title") or "chat" for it in items if it.get("id") == conn.chat_id), "chat"),
                    "turns": replay(conn.messages),
                    "messages": conn.messages,
                })
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass


async def wait_decision(conn: Conn, call_id: str) -> bool:
    while not conn.cancel.is_set():
        try:
            msg = await asyncio.wait_for(conn.inbox.get(), timeout=0.2)
        except asyncio.TimeoutError:
            continue
        if msg is None:
            conn.cancel.set()
            return False
        if msg.get("type") == "cancel":
            return False
        if msg.get("id") == call_id and "approved" in msg:
            return bool(msg.get("approved"))
    return False


async def stream_completion(conn: Conn, messages: list) -> tuple[str, list, bool]:
    """One streamed model call. Sends tokens as they arrive; returns the full
    text, any tool calls (reassembled from fragments), and whether we cancelled."""
    stream = await client.chat.completions.create(
        model=STATE["model"], messages=messages, tools=tools.TOOLS,
        temperature=TEMPERATURE, stream=True,
    )
    content = ""
    tcs: dict[int, dict] = {}
    started = False
    cancelled = False
    agen = stream.__aiter__()
    try:
        while True:
            if conn.cancel.is_set():
                cancelled = True
                break
            try:
                chunk = await asyncio.wait_for(agen.__anext__(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            except StopAsyncIteration:
                break
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                if not started:
                    await send(conn, {"type": "assistant_start"})
                    started = True
                content += delta.content
                await send(conn, {"type": "token", "content": delta.content})
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
    finally:
        aclose = getattr(stream, "aclose", None)
        if aclose:
            try:
                await aclose()
            except Exception:  # noqa: BLE001
                pass
    if started:
        await send(conn, {"type": "assistant_end"})
    ordered = [tcs[i] for i in sorted(tcs)]
    return content, ordered, cancelled


async def finish_cancelled(conn: Conn) -> None:
    await send(conn, {"type": "cancelled"})
    await send(conn, {"type": "status", "state": "idle"})


async def run_turn(conn: Conn) -> None:
    for m in conn.messages:
        if m.get("role") == "assistant" and m.get("content") is None:
            m["content"] = ""
    for _ in range(MAX_TOOL_HOPS):
        if conn.cancel.is_set():
            await finish_cancelled(conn)
            return
        await send(conn, {"type": "status", "state": "thinking"})
        try:
            content, tcs, cancelled = await stream_completion(conn, conn.messages)
        except Exception as e:  # noqa: BLE001
            await send(conn, {"type": "error", "content": str(e)})
            await send(conn, {"type": "status", "state": "idle"})
            return

        assistant_msg: dict = {"role": "assistant", "content": content or ""}
        if tcs:
            assistant_msg["tool_calls"] = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in tcs
            ]
        conn.messages.append(assistant_msg)

        if cancelled or conn.cancel.is_set():
            await finish_cancelled(conn)
            return

        if not tcs:
            await send(conn, {"type": "status", "state": "idle"})
            return

        for tc in tcs:
            if conn.cancel.is_set():
                await finish_cancelled(conn)
                return
            name = tc["name"]
            try:
                args = json.loads(tc["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}

            approved = True
            if name in tools.REQUIRES_APPROVAL:
                await send(conn, {"type": "tool_request", "id": tc["id"], "name": name, "args": args})
                approved = await wait_decision(conn, tc["id"])
                if conn.cancel.is_set():
                    result = "DENIED: user cancelled this turn."
                    await send(conn, {"type": "tool_result", "name": name, "content": result})
                    conn.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                    await finish_cancelled(conn)
                    return
            else:
                await send(conn, {"type": "tool_run", "name": name, "args": args})

            if not approved:
                result = "DENIED: user rejected this call."
            else:
                fn = tools.TOOL_FNS.get(name)
                result = await asyncio.to_thread(fn, **args) if fn else f"ERROR: unknown tool {name!r}"
                if name == "remember":
                    refresh_system_prompt(conn.messages)

            await send(conn, {"type": "tool_result", "name": name, "content": str(result)[:2000]})
            conn.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": str(result)})

            if conn.cancel.is_set():
                await finish_cancelled(conn)
                return

    await send(conn, {"type": "assistant", "content": "(hit tool-hop limit; stopping this turn)"})
    await send(conn, {"type": "status", "state": "idle"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
