# Qwen3.8-27B Abliterated NInfer on Runpod Serverless

This project packages NInfer for a scale-to-zero Runpod queue endpoint using:

- `lyf/Qwen3.8-27B-Huihui-Abliterated-NInfer-NVFP4`
- one GeForce RTX 5090 (32 GB)
- vision, thinking, and MTP speculative decoding
- a 204,800-token logical context ceiling with INT8 KV cache
- Runpod's durable job queue plus its OpenAI-compatible streaming gateway

## Runpod endpoint settings

Use the container image produced by `.github/workflows/container.yml`, then configure:

| Setting | Value |
|---|---|
| Endpoint type | Queue |
| GPU | GeForce RTX 5090 |
| GPUs per worker | 1 |
| Container disk | 80 GB |
| Active workers | 0 |
| Max workers | 1 |
| Idle timeout | 300 seconds |
| FlashBoot | Enabled |
| Cached model | `lyf/Qwen3.8-27B-Huihui-Abliterated-NInfer-NVFP4` |
| Internal NInfer port | `8080` |
| External OpenAI route | `https://api.runpod.ai/v2/ENDPOINT_ID/openai/v1` |

Copy the non-secret environment variables from `runpod.env.example`. Do not put a Runpod API key
or Hugging Face token in this repository. The model is public and does not require an HF token.

## Request example

```bash
curl https://api.runpod.ai/v2/ENDPOINT_ID/openai/v1/chat/completions \
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
Image URLs are preferable to base64 payloads for large images.

## Benchmark an active worker

The benchmark helper sends one warm-up followed by three measured 512-token requests:

```bash
python scripts/benchmark-ninfer.py \
  --base-url http://127.0.0.1:8080 \
  --runs 4 \
  --max-tokens 512
```

Add `--vision-file image.jpg` to benchmark an embedded image, or `--vision-url URL` to let the
worker fetch an image. Embedded images avoid certificate and remote-host variability.

## Operational notes

Scale-to-zero means the first request after idling must wait for a cold start and model load. Use
the asynchronous `/run` route for the first cold request so the job remains queued while Runpod
prepares the cached model; warm OpenAI requests can use `/openai/v1`. The 300-second idle timeout
keeps a short chat warm and then stops GPU billing. NInfer is compiled for `sm_120a` with a target
of 170 SMs, matching the RTX 5090.
