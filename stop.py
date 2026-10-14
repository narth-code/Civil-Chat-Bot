"""Stop what start.py started: the API and the UI (and with --all the Postgres container).

    python stop.py          stop the API and UI
    python stop.py --all    also `docker compose stop` (the data volume is never removed), stop Ollama (tray app,
                            server, model runners), any node/bun/esbuild process of this repository, and quit
                            Docker Desktop + `wsl --shutdown` (frees vmmemWSL; stops all WSL distros)

Only processes this repository started are touched. Leftovers from a crashed launcher are found by their command
line, which must contain this repository's path.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE_FILE = ROOT / ".run" / "state.json"
WINDOWS = os.name == "nt"
DOCKER_FALLBACK = Path("C:/Program Files/Docker/Docker/resources/bin/docker.exe")
# the API launched as `<repo>\.venv-ingest\Scripts\python.exe run.py`, and Vite started by `npm run dev`
API_COMMAND = re.compile(r"python(?:\.exe)?\"?\s+run\.py\s*$", re.IGNORECASE)
UI_COMMAND = re.compile(r"vite", re.IGNORECASE)
# with --all: JS runtimes/bundlers of this repo that outlive Vite, and Ollama (server + runner processes)
JS_RUNTIME = re.compile(r"(?:node|bun|esbuild)(?:\.exe)?\"?$", re.IGNORECASE)
OLLAMA_NAMES = ("ollama app", "ollama")
DOCKER_DESKTOP_NAMES = ("docker desktop", "com.docker")


def alive(pid: int) -> bool:
    if WINDOWS:
        listing = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in listing
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def kill_tree(pid: int) -> None:
    if WINDOWS:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except OSError:
        pass


def repo_processes() -> list[tuple[int, str]]:
    """(pid, command line) of running processes whose command line mentions this repository."""
    if not WINDOWS:
        listing = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout.splitlines()[1:]
        rows = [line.strip().split(None, 1) for line in listing]
        return [(int(pid), cmd) for pid, cmd in rows if len(pid) and str(ROOT) in cmd]
    script = "Get-CimInstance Win32_Process | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    output = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True).stdout
    try:
        rows = json.loads(output or "[]")
    except json.JSONDecodeError:
        return []
    rows = rows if isinstance(rows, list) else [rows]
    marker = str(ROOT).lower()
    return [
        (row["ProcessId"], row["CommandLine"])
        for row in rows
        if row.get("CommandLine") and marker in row["CommandLine"].lower() and row["ProcessId"] != os.getpid()
    ]


def named_processes(names: tuple[str, ...]) -> list[tuple[int, str]]:
    """(pid, name) of running processes whose executable name starts with one of `names` (case-insensitive)."""
    if WINDOWS:
        script = "Get-CimInstance Win32_Process | Select-Object ProcessId,Name | ConvertTo-Json -Compress"
        output = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True).stdout
        try:
            rows = json.loads(output or "[]")
        except json.JSONDecodeError:
            return []
        rows = rows if isinstance(rows, list) else [rows]
        pairs = [(row["ProcessId"], row["Name"]) for row in rows if row.get("Name")]
    else:
        listing = subprocess.run(["ps", "-eo", "pid,comm"], capture_output=True, text=True).stdout.splitlines()[1:]
        pairs = [(int(parts[0]), parts[1]) for parts in (line.split(None, 1) for line in listing) if len(parts) == 2]
    # the tray app first, otherwise it respawns the server
    return sorted(
        ((pid, name) for pid, name in pairs if name.lower().startswith(names) and pid != os.getpid()),
        key=lambda pair: " " not in pair[1],
    )


def find_docker() -> str | None:
    return shutil.which("docker") or (str(DOCKER_FALLBACK) if DOCKER_FALLBACK.exists() else None)


def stop_docker_desktop(docker: str) -> None:
    """Quit Docker Desktop and shut WSL down so `vmmemWSL` releases its memory (Windows only).

    `wsl --shutdown` stops every WSL distribution, not only Docker's.
    """
    if not WINDOWS:
        return
    subprocess.run([docker, "desktop", "stop"], capture_output=True, timeout=120)  # graceful; Docker Desktop 4.37+
    for pid, _ in named_processes(DOCKER_DESKTOP_NAMES):
        kill_tree(pid)
    subprocess.run(["wsl", "--shutdown"], capture_output=True, timeout=60)
    print("Docker Desktop quit and WSL shut down (start Docker Desktop again before `python start.py`).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stop the API and UI started by start.py.")
    parser.add_argument("--all", action="store_true", help="also stop the Postgres container")
    parser.add_argument(
        "--force", action="store_true", help="also stop any Vite dev server of this repo that start.py did not start"
    )
    args = parser.parse_args()

    stopped: list[str] = []
    state = json.loads(STATE_FILE.read_text(encoding="utf-8")) if STATE_FILE.exists() else {}
    for name in ("api", "ui"):
        service = state.get(name) or {}
        pid = service.get("pid")
        if service.get("started_by_script") and pid and alive(pid):
            kill_tree(pid)
            stopped.append(f"{name} (pid {pid})")

    done: set[int] = set()
    for pid, command in repo_processes():
        is_api = bool(API_COMMAND.search(command.strip()))
        is_stray_ui = args.force and "node" in command.lower() and bool(UI_COMMAND.search(command))
        if is_api or is_stray_ui:
            if alive(pid):
                kill_tree(pid)
                done.add(pid)
                stopped.append(f"leftover {command.split()[0].strip(chr(34))[-30:]} (pid {pid})")

    if args.all:
        for pid, command in repo_processes():
            if pid in done or not JS_RUNTIME.search(command.split()[0]) or not alive(pid):
                continue
            kill_tree(pid)
            done.add(pid)
            stopped.append(f"leftover {command.split()[0].strip(chr(34))[-30:]} (pid {pid})")
        for pid, name in named_processes(OLLAMA_NAMES):
            kill_tree(pid)
            stopped.append(f"{name} (pid {pid})")

    if STATE_FILE.exists():
        STATE_FILE.unlink()
    print("Stopped: " + ", ".join(stopped) if stopped else "Nothing of this repository was running.")

    if args.all:
        docker = find_docker()
        if docker is None:
            print("docker not found; Postgres not touched.")
        else:
            subprocess.run([docker, "compose", "stop"], cwd=ROOT)
            print("Postgres container stopped (data kept). Start it again with `python start.py`.")
            stop_docker_desktop(docker)
    else:
        print("Postgres left running (use `python stop.py --all` to stop it too).")


if __name__ == "__main__":
    main()
