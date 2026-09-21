"""
Tool implementations for the local agent.

Each tool DOES the thing and returns a string. It does not ask permission —
the interface (terminal or web server) handles that using REQUIRES_APPROVAL.

TOOLS is the JSON schema the model sees. TOOL_FNS maps name -> function.
"""

from __future__ import annotations

import fnmatch
import os
import subprocess
import urllib.request

# Where the agent keeps its durable note-to-self. Relative to where you run
# the server, i.e. your project folder.
MEMORY_FILE = "memory.md"

# Tools that can change your machine or reach the network. Interfaces confirm
# these before running. (remember/read_file/search are safe and auto-run.)
REQUIRES_APPROVAL = {"run_shell"}


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def remember(fact: str) -> str:
    """Append a durable fact to long-term memory (memory.md)."""
    fact = fact.strip()
    if not fact:
        return "Nothing to remember."
    try:
        with open(MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(f"- {fact}\n")
        return f"Saved to memory: {fact}"
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

def run_shell(command: str, timeout: int = 60) -> str:
    """Run a shell command and return combined stdout/stderr."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout,
        )
        out = (result.stdout or "") + (result.stderr or "")
        return out.strip()[:8000] or f"(no output, exit code {result.returncode})"
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {timeout}s"
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Local files
# ---------------------------------------------------------------------------

def read_file(path: str, max_chars: int = 8000) -> str:
    """Read a text file from disk."""
    try:
        with open(os.path.expanduser(path), "r", encoding="utf-8", errors="replace") as f:
            data = f.read(max_chars + 1)
        if len(data) > max_chars:
            return data[:max_chars] + f"\n... [truncated at {max_chars} chars]"
        return data or "(empty file)"
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


def search_files(pattern: str, directory: str = ".", contains: str = "") -> str:
    """Find files matching a glob pattern, optionally containing a string."""
    matches = []
    root = os.path.expanduser(directory)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".git", "node_modules", "__pycache__", ".venv"}]
        for name in filenames:
            if not fnmatch.fnmatch(name, pattern):
                continue
            full = os.path.join(dirpath, name)
            if contains:
                try:
                    with open(full, "r", encoding="utf-8", errors="ignore") as f:
                        if contains not in f.read():
                            continue
                except Exception:  # noqa: BLE001
                    continue
            matches.append(full)
            if len(matches) >= 100:
                break
    return "\n".join(matches) if matches else "(no matches)"


# ---------------------------------------------------------------------------
# Web
# ---------------------------------------------------------------------------

def fetch_url(url: str, max_chars: int = 6000) -> str:
    """Fetch a URL and return its raw text (no JS rendering)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "local-agent/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        return body[:max_chars] + ("\n... [truncated]" if len(body) > max_chars else "")
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Save a durable fact about the user or their setup to long-term memory. Use when you learn something worth recalling in future sessions (preferences, names, paths, ongoing projects). Do not save trivia or one-off details.",
            "parameters": {
                "type": "object",
                "properties": {"fact": {"type": "string", "description": "The fact to remember, as a short sentence."}},
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command on the local machine and return its output.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "The shell command to run."}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a local text file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path to the file."}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Recursively find files by glob pattern, optionally filtering to those containing a string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob, e.g. '*.py'."},
                    "directory": {"type": "string", "description": "Directory to search. Defaults to '.'."},
                    "contains": {"type": "string", "description": "Optional substring the file must contain."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch the text content of a web page by URL.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "The URL to fetch."}},
                "required": ["url"],
            },
        },
    },
]

TOOL_FNS = {
    "remember": remember,
    "run_shell": run_shell,
    "read_file": read_file,
    "search_files": search_files,
    "fetch_url": fetch_url,
}
