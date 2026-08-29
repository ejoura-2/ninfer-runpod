#!/usr/bin/env python3
"""Small OpenAI-compatible NInfer throughput benchmark."""

import argparse
import base64
import json
from pathlib import Path
import time
import urllib.request


def request(base_url: str, model: str, max_tokens: int, vision_url: str | None) -> dict:
    content: str | list[dict[str, object]]
    if vision_url:
        content = [
            {"type": "text", "text": "Describe this image thoroughly, then reason about what is happening."},
            {"type": "image_url", "image_url": {"url": vision_url}},
        ]
    else:
        content = (
            "Write exactly 400 numbered, very short facts about mathematics. "
            "Continue until all 400 are complete; do not add an introduction."
        )
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    encoded = json.dumps(body).encode()
    started = time.perf_counter()
    with urllib.request.urlopen(
        urllib.request.Request(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            data=encoded,
            headers={"Content-Type": "application/json"},
        ),
        timeout=600,
    ) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - started
    usage = result.get("usage", {})
    completion_tokens = usage.get("completion_tokens") or 0
    return {
        "elapsed_seconds": round(elapsed, 4),
        "completion_tokens": completion_tokens,
        "wall_tokens_per_second": round(completion_tokens / elapsed, 2),
        "finish_reason": result.get("choices", [{}])[0].get("finish_reason"),
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="qwen3.8-27b-huihui-abliterated")
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--vision-url")
    parser.add_argument("--vision-file", type=Path)
    args = parser.parse_args()

    vision_url = args.vision_url
    if args.vision_file:
        encoded = base64.b64encode(args.vision_file.read_bytes()).decode()
        vision_url = f"data:image/jpeg;base64,{encoded}"

    for index in range(args.runs):
        result = request(args.base_url, args.model, args.max_tokens, vision_url)
        result["run"] = index + 1
        result["warmup"] = index == 0
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
