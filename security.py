"""Shared safety checks for tools and the web server.

File access is limited to the process working directory. Fetch is limited to
public http(s). Shell is off unless AGENT_ALLOW_SHELL=1; even then a deny-list
is only extra friction, not a sandbox.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlparse

PROJECT_ROOT = os.path.realpath(os.getcwd())
MEMORY_MAX = 32 * 1024
MAX_WALK_DEPTH = 8
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "chats", "env"}
DENIED_NAMES = {".agent_secret", ".env"}

SHELL_DENY = (
    "rm -rf /",
    "rm -rf /*",
    "rm -fr /",
    "format ",
    "mkfs",
    "del /s",
    "rd /s",
    "rmdir /s",
    "shutdown",
    "reg delete",
    "bcdedit",
    "diskpart",
    ":(){ :|:& };:",
    "cipher /w",
    "dd if=",
)


def allow_shell() -> bool:
    return os.environ.get("AGENT_ALLOW_SHELL", "").strip().lower() in {"1", "true", "yes", "on"}


def sanitize_text(text: str) -> str:
    return "".join(c for c in text if c in "\n\t" or ord(c) >= 32)


def resolve_in_project(path: str) -> str | None:
    """Return a real path inside PROJECT_ROOT, or None if the path is forbidden."""
    raw = os.path.expanduser(path or "")
    if not raw:
        return None
    if not os.path.isabs(raw):
        raw = os.path.join(PROJECT_ROOT, raw)
    real = os.path.realpath(raw)
    root = PROJECT_ROOT
    if real != root and not real.startswith(root + os.sep):
        return None
    if os.path.basename(real) in DENIED_NAMES:
        return None
    rel = os.path.relpath(real, root)
    if rel != "." and ".git" in rel.split(os.sep):
        return None
    return real


def rel_from_project(path: str) -> str:
    return os.path.relpath(path, PROJECT_ROOT)


def walk_depth(root: str, dirpath: str) -> int:
    rel = os.path.relpath(dirpath, root)
    if rel == ".":
        return 0
    return rel.count(os.sep) + 1


def shell_denied(command: str) -> str | None:
    low = command.lower()
    for pat in SHELL_DENY:
        if pat.lower() in low:
            return pat
    return None


def _ip_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if ip.is_private or ip.is_loopback or ip.is_link_local:
        return True
    if ip.is_multicast or ip.is_reserved or ip.is_unspecified:
        return True
    if str(ip) in {"169.254.169.254", "::1"}:
        return True
    return False


def url_blocked(url: str) -> str | None:
    """None if the URL may be fetched; otherwise an ERROR string."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return "ERROR: only http/https URLs are allowed"
    host = parsed.hostname
    if not host:
        return "ERROR: invalid URL"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return "ERROR: could not resolve host"
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if _ip_blocked(ip):
            return "ERROR: URL points to a private or local address"
    return None
