#!/usr/bin/env python3
"""Start NInfer, wait for readiness, then join the Runpod job queue."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time

import entrypoint


HEALTH_POLL_INTERVAL = 2.0
ninfer_process: subprocess.Popen | None = None


def wait_for_ninfer(process: subprocess.Popen, port: int, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"ninfer-serve exited during startup with code {process.returncode}"
            )
        if entrypoint.upstream_ready(port):
            logging.info("NInfer is healthy")
            return
        time.sleep(HEALTH_POLL_INTERVAL)
    raise RuntimeError(f"NInfer did not become healthy within {timeout}s")


def _forward_signal(signum: int, _frame: object) -> None:
    if ninfer_process and ninfer_process.poll() is None:
        ninfer_process.send_signal(signum)
    raise SystemExit(128 + signum)


def main() -> int:
    global ninfer_process

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    model_path = entrypoint.resolve_model_path()
    model_port = entrypoint.env_int("PORT", 8080)
    timeout = entrypoint.env_int("RUNPOD_INIT_TIMEOUT", 1200)
    command = entrypoint.build_command(model_path)
    api_key = os.getenv("NINFER_API_KEY")
    safe_command = ["<redacted>" if api_key and arg == api_key else arg for arg in command]
    logging.info("Starting: %s", " ".join(safe_command))

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _forward_signal)

    ninfer_process = subprocess.Popen(
        command,
        env=entrypoint.runtime_environment(),
    )
    try:
        wait_for_ninfer(ninfer_process, model_port, timeout)
    except RuntimeError as exc:
        logging.error("%s", exc)
        return 1

    import queue_handler
    import runpod

    queue_handler.ninfer_process = ninfer_process
    max_concurrency = entrypoint.env_int("MAX_CONCURRENCY", 1)
    runpod.serverless.start(
        {
            "handler": queue_handler.handler,
            "concurrency_modifier": lambda _current: max_concurrency,
            "return_aggregate_stream": True,
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
