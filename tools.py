"""
Tool implementations for the local agent.

Each tool DOES the thing and returns a string. It does not ask permission —
the interface (terminal or web server) handles that using REQUIRES_APPROVAL.

File tools are sandboxed to the process working directory. fetch_url is
approval-gated and blocked from private/local addresses. run_shell is omitted
unless AGENT_ALLOW_SHELL=1.

TOOLS is the JSON schema the model sees. TOOL_FNS maps name -> function.
"""

from __future__ import annotations

import fnmatch
import os
import subprocess
import urllib.request

import security

# Where the agent keeps its durable note-to-self. Relative to where you run
# the server, i.e. your project folder.
MEMORY_FILE = "memory.md"

REQUIRES_APPROVAL = {"fetch_url"}
if security.allow_shell():
    REQUIRES_APPROVAL.add("run_shell")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ARG002
        return None


_FETCH_OPENER = urllib.request.build_opener(_NoRedirect)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def remember(fact: str) -> str:
    """Append a durable fact to long-term memory (memory.md)."""
    fact = security.sanitize_text(fact).strip()
    if not fact:
        return "Nothing to remember."
    line = f"- {fact}\n"
    try:
        existing = ""
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, encoding="utf-8") as f:
                existing = f.read()
        if len(existing.encode("utf-8")) + len(line.encode("utf-8")) > security.MEMORY_MAX:
            return "ERROR: memory is full (32 KiB cap)."
        with open(MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(line)
        return f"Saved to memory: {fact}"
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


def write_memory_text(content: str) -> str:
    """Overwrite memory.md, enforcing size and character limits. Returns stored text."""
    text = security.sanitize_text(content)
    raw = text.encode("utf-8")
    if len(raw) > security.MEMORY_MAX:
        text = raw[: security.MEMORY_MAX].decode("utf-8", errors="ignore")
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    return text


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

def run_shell(command: str, timeout: int = 60) -> str:
    """Run a shell command and return combined stdout/stderr."""
    if not security.allow_shell():
        return "ERROR: shell is disabled. Set AGENT_ALLOW_SHELL=1 to enable."
    hit = security.shell_denied(command)
    if hit:
        return f"ERROR: command blocked by deny-list (matched {hit!r})."
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
    """Read a text file from disk (project folder only)."""
    real = security.resolve_in_project(path)
    if not real:
        return "ERROR: path is outside the project folder or is denied."
    if not os.path.isfile(real):
        return "ERROR: not a file"
    try:
        with open(real, "r", encoding="utf-8", errors="replace") as f:
            data = f.read(max_chars + 1)
        if len(data) > max_chars:
            return data[:max_chars] + f"\n... [truncated at {max_chars} chars]"
        return data or "(empty file)"
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


def search_files(pattern: str, directory: str = ".", contains: str = "") -> str:
    """Find files matching a glob pattern, optionally containing a string."""
    real_root = security.resolve_in_project(directory or ".")
    if not real_root:
        return "ERROR: directory is outside the project folder or is denied."
    if not os.path.isdir(real_root):
        return "ERROR: not a directory"
    matches: list[str] = []
    for dirpath, dirnames, filenames in os.walk(real_root):
        dirnames[:] = [d for d in dirnames if d not in security.SKIP_DIRS]
        if security.walk_depth(real_root, dirpath) >= security.MAX_WALK_DEPTH:
            dirnames.clear()
        for name in filenames:
            if name in security.DENIED_NAMES:
                continue
            if not fnmatch.fnmatch(name, pattern):
                continue
            full = os.path.join(dirpath, name)
            allowed = security.resolve_in_project(full)
            if not allowed:
                continue
            if contains:
                try:
                    with open(allowed, "r", encoding="utf-8", errors="ignore") as f:
                        if contains not in f.read():
                            continue
                except Exception:  # noqa: BLE001
                    continue
            matches.append(security.rel_from_project(allowed))
            if len(matches) >= 100:
                break
        if len(matches) >= 100:
            break
    return "\n".join(matches) if matches else "(no matches)"


# ---------------------------------------------------------------------------
# Web
# ---------------------------------------------------------------------------

def fetch_url(url: str, max_chars: int = 6000) -> str:
    """Fetch a public http(s) URL and return its raw text (no JS rendering)."""
    blocked = security.url_blocked(url)
    if blocked:
        return blocked
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "local-agent/1.0"})
        with _FETCH_OPENER.open(req, timeout=20) as resp:
            final = resp.geturl()
            blocked = security.url_blocked(final)
            if blocked:
                return blocked
            body = resp.read().decode("utf-8", errors="replace")
        return body[:max_chars] + ("\n... [truncated]" if len(body) > max_chars else "")
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REMEMBER = {
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
}
_RUN_SHELL = {
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
}
_READ_FILE = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a text file inside the project folder.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to the file, relative to the project folder."}},
            "required": ["path"],
        },
    },
}
_SEARCH = {
    "type": "function",
    "function": {
        "name": "search_files",
        "description": "Recursively find files by glob pattern inside the project folder, optionally filtering to those containing a string.",
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
}
_FETCH = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": "Fetch the text content of a public web page by URL (requires user approval).",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "The URL to fetch."}},
            "required": ["url"],
        },
    },
}

TOOLS = [_REMEMBER, _READ_FILE, _SEARCH, _FETCH]
TOOL_FNS = {
    "remember": remember,
    "read_file": read_file,
    "search_files": search_files,
    "fetch_url": fetch_url,
}
if security.allow_shell():
    TOOLS.insert(1, _RUN_SHELL)
    TOOL_FNS["run_shell"] = run_shell
