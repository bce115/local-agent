# local agent

A small local UI for chatting with models served by [Ollama](https://ollama.com). It streams replies, keeps named chats on disk, and can read files in this folder, remember facts, and (optionally) run shell commands.

```text
pip install -r requirements.txt
ollama pull devstral-small-2
python server.py
```

Then open `http://127.0.0.1:8000/login`. On first run a password is written to `.agent_secret` and printed in the terminal.

The terminal client is `python agent.py` (same tools, no HTTP).

## Security

Defaults are **read-mostly** and **loopback**:

- Login is always required (cookie session).
- File tools only see the **project folder** (the process working directory). Paths like `C:\Users\...` or `~` are rejected. `.git`, `.agent_secret`, and `.env` are denied. Search results are relative paths and skip `chats/`.
- `fetch_url` needs **Allow** in the UI and refuses `file://`, loopback, link-local, and RFC1918 addresses (including `169.254.169.254`). Redirects are not followed.
- **`run_shell` is off** unless you set `AGENT_ALLOW_SHELL=1`. The model will not even see the tool.

### LAN

Other devices on your Wi-Fi can use the UI only if you bind a non-loopback address **and** a password exists (`AGENT_PASSWORD` or `.agent_secret`):

```text
set AGENT_HOST=0.0.0.0
set AGENT_PASSWORD=choose-a-long-secret
python server.py
```

Open the machine’s LAN IP on port 8000, sign in, then use the app. Do not open the port on a router/firewall unless you intend that. There is **no TLS** in this project: on a hostile network someone can read the password. Use a trusted LAN or a VPN.

The server refuses a non-loopback bind if no password is configured.

### Enabling shell

```text
set AGENT_ALLOW_SHELL=1
python server.py
```

Shell still goes through the Allow/Deny gate. A deny-list blocks some obvious destructive strings (`rm -rf /`, `format`, `shutdown`, …). **That list is incomplete** and is not a sandbox. A determined model or user can still propose harmful commands; you are the last check. Leave shell off unless you need it.

### Do not commit

`.gitignore` already excludes `.agent_secret`, `.env`, `memory.md`, `chats/`, and `history.json`. Those hold secrets and personal transcripts.
