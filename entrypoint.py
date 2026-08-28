#!/usr/bin/env python3
"""Run NInfer and expose Runpod's required readiness endpoint."""

from __future__ import annotations

import glob
import http.client
import os
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise SystemExit(f"{name} must be at least {minimum}, got {value}")
    return value


def resolve_model_path() -> Path:
    explicit = os.getenv("MODEL_PATH")
    filename = os.getenv("MODEL_FILENAME", "qwen3_8_27b_nvfp4.ninfer")

    if explicit:
        candidates = [Path(explicit)]
    else:
        repo_id = os.getenv(
            "MODEL_REPO_ID",
            "lyf/Qwen3.8-27B-Huihui-Abliterated-NInfer-NVFP4",
        )
        cache_name = "models--" + repo_id.replace("/", "--")
        patterns = [
            f"/runpod-volume/huggingface-cache/hub/{cache_name}/snapshots/*/{filename}",
            f"/runpod-volume/{filename}",
            f"/models/{filename}",
        ]
        candidates = [Path(path) for pattern in patterns for path in glob.glob(pattern)]

    files = [path for path in candidates if path.is_file()]
    if not files:
        searched = explicit or (
            "/runpod-volume/huggingface-cache/hub/<cached-model>/snapshots/*/"
            + filename
        )
        raise SystemExit(
            "NInfer model artifact was not found. Configure the endpoint cached model as "
            f"{os.getenv('MODEL_REPO_ID')} or set MODEL_PATH. Searched: {searched}"
        )
    return max(files, key=lambda path: path.stat().st_mtime)


def build_command(model_path: Path) -> list[str]:
    port = env_int("PORT", 8080)
    command = [
        os.getenv("NINFER_BIN", "ninfer-serve"),
        str(model_path),
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--model-id",
        os.getenv("MODEL_ID", "qwen3.8-27b-huihui-abliterated"),
        "--max-context",
        str(env_int("MAX_CONTEXT", 262144)),
        "--kv-capacity",
        os.getenv("KV_CAPACITY", "262144"),
        "--max-concurrency",
        str(env_int("MAX_CONCURRENCY", 1)),
        "--prefill-chunk",
        str(env_int("PREFILL_CHUNK", 4096)),
        "--kv-dtype",
        os.getenv("KV_DTYPE", "int8"),
        "--default-max-tokens",
        str(env_int("DEFAULT_MAX_TOKENS", 32768)),
        "--max-request-mib",
        str(env_int("MAX_REQUEST_MIB", 28)),
        "--vision",
        "--spec",
        "mtp",
        "--draft-tokens",
        str(env_int("DRAFT_TOKENS", 3)),
        "--lm-head-draft",
        "--preserve-thinking",
    ]
    api_key = os.getenv("NINFER_API_KEY")
    if api_key:
        command.extend(["--api-key", api_key])
    return command


def upstream_ready(port: int) -> bool:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
    try:
        connection.request("GET", "/health")
        response = connection.getresponse()
        response.read()
        return 200 <= response.status < 300
    except OSError:
        return False
    finally:
        connection.close()


def make_health_handler(model_port: int):
    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path != "/ping":
                self.send_error(404)
                return
            self.send_response(200 if upstream_ready(model_port) else 204)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, _format: str, *args: object) -> None:
            return

    return HealthHandler


def main() -> int:
    model_path = resolve_model_path()
    model_port = env_int("PORT", 8080)
    health_port = env_int("PORT_HEALTH", 8081)
    if model_port == health_port:
        raise SystemExit("PORT and PORT_HEALTH must be different")

    server = ThreadingHTTPServer(
        ("0.0.0.0", health_port), make_health_handler(model_port)
    )
    health_thread = threading.Thread(target=server.serve_forever, daemon=True)
    health_thread.start()

    command = build_command(model_path)
    safe_command = ["<redacted>" if arg == os.getenv("NINFER_API_KEY") else arg for arg in command]
    print("Starting:", " ".join(safe_command), flush=True)
    process = subprocess.Popen(command)

    def forward_signal(signum: int, _frame: object) -> None:
        if process.poll() is None:
            process.send_signal(signum)

    signal.signal(signal.SIGTERM, forward_signal)
    signal.signal(signal.SIGINT, forward_signal)
    try:
        return process.wait()
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    sys.exit(main())
