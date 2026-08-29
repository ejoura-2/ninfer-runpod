# Qwen3.8-27B Abliterated NInfer on Runpod Serverless

This project packages NInfer for a scale-to-zero Runpod load-balancing endpoint using:

- `lyf/Qwen3.8-27B-Huihui-Abliterated-NInfer-NVFP4`
- one RTX PRO 6000 Blackwell Server Edition (96 GB)
- vision, thinking, and MTP speculative decoding
- a 262,144-token logical context ceiling with INT8 KV cache
- an OpenAI-compatible streaming API

## Runpod endpoint settings

Use the container image produced by `.github/workflows/container.yml`, then configure:

| Setting | Value |
|---|---|
| Endpoint type | Load Balancer |
| GPU | RTX PRO 6000 Blackwell Server Edition |
| GPUs per worker | 1 |
| Container disk | 40 GB |
| Active workers | 0 |
| Max workers | 1 |
| Idle timeout | 60 seconds |
| FlashBoot | Enabled |
| Cached model | `lyf/Qwen3.8-27B-Huihui-Abliterated-NInfer-NVFP4` |
| HTTP port | `8080` |
| OpenAI API port | `8080` |
| Readiness port | `8081` |
| Health path | `/ping` |

Copy the non-secret environment variables from `runpod.env.example`. Do not put a Runpod API key
or Hugging Face token in this repository. The model is public and does not require an HF token.

## Request example

```bash
curl https://ENDPOINT_ID.api.runpod.ai/v1/chat/completions \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b-huihui-abliterated",
    "messages": [{"role": "user", "content": "Think carefully and explain KV caching."}],
    "reasoning_effort": "xhigh",
    "max_tokens": 8192,
    "stream": true
  }'
```

For vision, use OpenAI-compatible typed content with an `image_url` HTTP(S) URL or base64 data URL.
Load-balancing requests have a 30 MB payload limit, so URLs are preferable for large images.

## Operational notes

Scale-to-zero means the first request after idling must wait for a cold start and model load. The
60-second idle timeout avoids repeatedly unloading the model during a short chat while still
stopping billing shortly after use. NInfer is compiled for `sm_120a` and this image applies
`patches/ninfer-target-sm-count.patch` with a target of 188 SMs, matching the RTX PRO 6000
Blackwell Server Edition. The upstream default remains 170 SMs for RTX 5090 builds.
