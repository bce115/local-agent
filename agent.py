"""
A minimal local AI agent (terminal version).

Runtime : Ollama serving a model on localhost:11434 (OpenAI-compatible API)
Brain   : whatever model you `ollama pull` (default: gemma4:12b)
Body    : the tools in tools.py (shell, file read, file search, web fetch)

The confirmation gate now lives HERE, in the loop, not inside the tool. The tool
just does the work; this interface decides when to ask you first (using
tools.REQUIRES_APPROVAL). The web server does the same job its own way.

    pip install openai
    ollama pull gemma4:12b
    python agent.py
"""

from __future__ import annotations

import json

from openai import OpenAI

import tools

MODEL = "gemma4:12b"
BASE_URL = "http://localhost:11434/v1"
TEMPERATURE = 0.7
MAX_TOOL_HOPS = 8

# Ask before running an approval-gated tool. Leave on while you're building.
CONFIRM = True

SYSTEM_PROMPT = """You are a local assistant running on the user's own machine.
You have tools to run shell commands, read and search local files, and fetch
web pages. Use them when they help; answer directly when they don't. Be concise
and technical. When you run a command, briefly say why before you call it."""

client = OpenAI(base_url=BASE_URL, api_key="ollama")


def approve(name: str, args: dict) -> bool:
    if not (CONFIRM and name in tools.REQUIRES_APPROVAL):
        return True
    print(f"\n  \033[33m! agent wants to run {name}:\033[0m {args}")
    return input("  allow? [y/N] ").strip().lower() == "y"


def run() -> None:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print(f"local agent  ·  {MODEL}  ·  ctrl-c to quit\n")

    while True:
        try:
            user = input("\033[36myou›\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        messages.append({"role": "user", "content": user})

        for _ in range(MAX_TOOL_HOPS):
            resp = client.chat.completions.create(
                model=MODEL, messages=messages, tools=tools.TOOLS, temperature=TEMPERATURE,
            )
            msg = resp.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                print(f"\033[32magent›\033[0m {msg.content}\n")
                break

            for call in msg.tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                if not approve(name, args):
                    result = "DENIED: user rejected this call."
                else:
                    fn = tools.TOOL_FNS.get(name)
                    result = fn(**args) if fn else f"ERROR: unknown tool {name!r}"
                messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})
        else:
            print("\033[31m(hit tool-hop limit; stopping this turn)\033[0m\n")


if __name__ == "__main__":
    run()
