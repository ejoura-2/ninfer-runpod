#!/usr/bin/env python3
"""Runpod queue handler that proxies jobs to the local NInfer HTTP server."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterable
from typing import Any, AsyncGenerator

import aiohttp


NINFER_PORT = os.getenv("PORT", "8080")
NINFER_BASE_URL = os.getenv(
    "NINFER_BASE_URL", f"http://127.0.0.1:{NINFER_PORT}"
)
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "3600"))
DEFAULT_CHAT_ROUTE = "/v1/chat/completions"

ninfer_process = None


def _is_ninfer_alive() -> bool:
    return ninfer_process is None or ninfer_process.poll() is None


def normalize_job_input(job_input: dict[str, Any]) -> tuple[str, str, dict | None]:
    """Return a local NInfer route, HTTP method, and optional request body."""
    if job_input.get("openai_input"):
        route = job_input.get("openai_route") or DEFAULT_CHAT_ROUTE
        return route, "POST", job_input["openai_input"]

    if "openai_route" in job_input:
        return job_input["openai_route"], "GET", None

    if "route" in job_input:
        route = job_input["route"]
        body = job_input.get("body")
        method = (job_input.get("method") or ("POST" if body is not None else "GET")).upper()
        return route, method, body

    messages = job_input.get("messages")
    prompt = job_input.get("prompt")
    if messages is None and prompt is None:
        raise ValueError(
            "Job input must contain openai_input, route/body, prompt, or messages"
        )

    body = dict(job_input.get("sampling_params") or {})
    body["stream"] = bool(job_input.get("stream", False))
    body.setdefault("model", os.getenv("MODEL_ID", "qwen3.8-27b-huihui-abliterated"))
    if messages is not None:
        body["messages"] = messages
        return DEFAULT_CHAT_ROUTE, "POST", body

    # NInfer intentionally exposes Chat Completions rather than the legacy
    # OpenAI /v1/completions route. Preserve the shorthand by promoting the
    # prompt to a user message.
    body["messages"] = [{"role": "user", "content": prompt}]
    return DEFAULT_CHAT_ROUTE, "POST", body


async def iter_sse_frames(chunks: AsyncIterable[bytes]) -> AsyncGenerator[str, None]:
    """Yield complete SSE events regardless of upstream TCP chunk boundaries.

    Runpod serializes each handler yield as one streamed output item. NInfer can
    split one SSE event across reads or coalesce the terminal finish_reason and
    [DONE] events into one read, so forwarding ``iter_any()`` chunks verbatim can
    make OpenAI clients miss the terminal finish_reason. Reframing here keeps the
    wire payload unchanged while making every yield one complete SSE event.
    """
    buffer = bytearray()
    async for chunk in chunks:
        if not chunk:
            continue
        buffer.extend(chunk)
        while True:
            lf_boundary = buffer.find(b"\n\n")
            crlf_boundary = buffer.find(b"\r\n\r\n")
            boundaries = [
                (index, size)
                for index, size in ((lf_boundary, 2), (crlf_boundary, 4))
                if index >= 0
            ]
            if not boundaries:
                break
            index, size = min(boundaries, key=lambda item: item[0])
            end = index + size
            frame = bytes(buffer[:end])
            del buffer[:end]
            yield frame.decode("utf-8", errors="replace")

    # SSE permits dispatching the final event at EOF even without a blank-line
    # separator. Forward it rather than silently discarding a useful error.
    if buffer:
        yield bytes(buffer).decode("utf-8", errors="replace")


def _error(message: str) -> dict[str, dict[str, str | None]]:
    return {
        "error": {
            "message": message,
            "type": "worker_error",
            "code": None,
        }
    }


async def handler(job: dict[str, Any]) -> AsyncGenerator[Any, None]:
    job_input = job.get("input") or {}
    try:
        route, method, body = normalize_job_input(job_input)
    except ValueError as exc:
        yield _error(str(exc))
        return

    if not isinstance(route, str) or not route.startswith("/"):
        yield _error("Proxy route must be an absolute HTTP path")
        return
    if not _is_ninfer_alive():
        yield _error("NInfer server process is not running")
        return

    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("NINFER_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    try:
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.request(
                method, f"{NINFER_BASE_URL}{route}", json=body
            ) as response:
                if response.status >= 400:
                    detail = await response.text()
                    logging.error(
                        "NInfer %s %s returned HTTP %s: %s",
                        method,
                        route,
                        response.status,
                        detail,
                    )
                    yield _error(f"NInfer returned HTTP {response.status}: {detail}")
                    return

                wants_stream = isinstance(body, dict) and body.get("stream") is True
                if wants_stream:
                    async for frame in iter_sse_frames(response.content.iter_any()):
                        yield frame
                else:
                    yield await response.json(content_type=None)
    except (aiohttp.ClientError, TimeoutError) as exc:
        logging.exception("Request to NInfer failed")
        yield _error(f"Request to NInfer failed: {exc}")
