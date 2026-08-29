import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp import web


SPEC = importlib.util.spec_from_file_location(
    "queue_handler", Path(__file__).parents[1] / "queue_handler.py"
)
queue_handler = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(queue_handler)


class QueueHandlerTests(unittest.TestCase):
    def test_openai_passthrough_shape(self):
        body = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
        route, method, normalized = queue_handler.normalize_job_input(
            {"openai_route": "/v1/chat/completions", "openai_input": body}
        )
        self.assertEqual(route, "/v1/chat/completions")
        self.assertEqual(method, "POST")
        self.assertEqual(normalized, body)
        self.assertIsNot(normalized, body)

    def test_pi_reasoning_effort_high_maps_to_ninfer_xhigh(self):
        body = {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "high",
        }
        _, _, normalized = queue_handler.normalize_job_input(
            {"openai_route": "/v1/chat/completions", "openai_input": body}
        )
        self.assertEqual(normalized["reasoning_effort"], "xhigh")
        self.assertEqual(body["reasoning_effort"], "high")

    def test_openai_reasoning_effort_aliases_map_to_supported_levels(self):
        for source, expected in {
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "xhigh",
            "xhigh": "xhigh",
            "max": "xhigh",
            "none": "none",
        }.items():
            with self.subTest(source=source):
                normalized = queue_handler.normalize_reasoning_effort(
                    {"reasoning_effort": source}
                )
                self.assertEqual(normalized["reasoning_effort"], expected)

    def test_openai_models_route_is_get(self):
        route, method, body = queue_handler.normalize_job_input(
            {"openai_route": "/v1/models"}
        )
        self.assertEqual((route, method, body), ("/v1/models", "GET", None))

    def test_openai_models_route_with_empty_input_is_get(self):
        route, method, body = queue_handler.normalize_job_input(
            {"openai_route": "/v1/models", "openai_input": {}}
        )
        self.assertEqual((route, method, body), ("/v1/models", "GET", None))

    def test_legacy_chat_adds_model_and_sampling_parameters(self):
        with patch.dict(os.environ, {"MODEL_ID": "served-model"}, clear=True):
            route, method, body = queue_handler.normalize_job_input(
                {
                    "messages": [{"role": "user", "content": "hi"}],
                    "sampling_params": {"max_tokens": 8},
                }
            )
        self.assertEqual(route, "/v1/chat/completions")
        self.assertEqual(method, "POST")
        self.assertEqual(body["model"], "served-model")
        self.assertEqual(body["max_tokens"], 8)
        self.assertFalse(body["stream"])

    def test_legacy_prompt_uses_ninfer_chat_route(self):
        with patch.dict(os.environ, {"MODEL_ID": "served-model"}, clear=True):
            route, method, body = queue_handler.normalize_job_input(
                {"prompt": "hi", "sampling_params": {"max_tokens": 8}}
            )
        self.assertEqual(route, "/v1/chat/completions")
        self.assertEqual(method, "POST")
        self.assertEqual(
            body["messages"], [{"role": "user", "content": "hi"}]
        )
        self.assertNotIn("prompt", body)

    def test_invalid_job_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must contain"):
            queue_handler.normalize_job_input({})


class SSEFrameTests(unittest.IsolatedAsyncioTestCase):
    async def test_sse_frames_are_reassembled_and_split(self):
        async def chunks():
            yield b'data: {"choices":[{"delta":{"content":"h"},"finish_reason":null}]}\n\n'
            yield b'data: {"choices":[{"delta":{},"finish_'
            yield b'reason":"stop"}]}\n\ndata: [DO'
            yield b'NE]\n\n'

        output = [frame async for frame in queue_handler.iter_sse_frames(chunks())]
        self.assertEqual(
            output,
            [
                'data: {"choices":[{"delta":{"content":"h"},"finish_reason":null}]}\n\n',
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
                "data: [DONE]\n\n",
            ],
        )

    async def test_sse_frames_accept_crlf_and_flush_trailing_data(self):
        async def chunks():
            yield b"event: message\r\ndata: one\r\n\r\ndata: trailing"

        output = [frame async for frame in queue_handler.iter_sse_frames(chunks())]
        self.assertEqual(output, ["event: message\r\ndata: one\r\n\r\n", "data: trailing"])


class QueueProxyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        async def chat(request):
            body = await request.json()
            if body.get("stream"):
                response = web.StreamResponse(
                    status=200, headers={"Content-Type": "text/event-stream"}
                )
                await response.prepare(request)
                await response.write(
                    b'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
                )
                await response.write(
                    b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
                )
                await response.write_eof()
                return response
            return web.json_response(
                {
                    "id": "test-response",
                    "model": body["model"],
                    "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                }
            )

        self.app = web.Application()
        self.app.router.add_post("/v1/chat/completions", chat)
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        port = self.site._server.sockets[0].getsockname()[1]
        self.previous_base_url = queue_handler.NINFER_BASE_URL
        queue_handler.NINFER_BASE_URL = f"http://127.0.0.1:{port}"

    async def asyncTearDown(self):
        queue_handler.NINFER_BASE_URL = self.previous_base_url
        await self.runner.cleanup()

    async def test_openai_request_is_proxied_to_local_server(self):
        output = [
            item
            async for item in queue_handler.handler(
                {
                    "input": {
                        "openai_route": "/v1/chat/completions",
                        "openai_input": {
                            "model": "served-model",
                            "messages": [{"role": "user", "content": "hi"}],
                        },
                    }
                }
            )
        ]
        self.assertEqual(output[0]["model"], "served-model")
        self.assertEqual(output[0]["choices"][0]["message"]["content"], "hi")

    async def test_streaming_request_yields_complete_openai_sse_frames(self):
        output = [
            item
            async for item in queue_handler.handler(
                {
                    "input": {
                        "openai_route": "/v1/chat/completions",
                        "openai_input": {
                            "model": "served-model",
                            "messages": [{"role": "user", "content": "hi"}],
                            "stream": True,
                        },
                    }
                }
            )
        ]
        self.assertEqual(len(output), 3)
        self.assertIn('"finish_reason":"stop"', output[1])
        self.assertEqual(output[2], "data: [DONE]\n\n")


if __name__ == "__main__":
    unittest.main()
