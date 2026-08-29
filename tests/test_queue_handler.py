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
        self.assertIs(normalized, body)

    def test_openai_models_route_is_get(self):
        route, method, body = queue_handler.normalize_job_input(
            {"openai_route": "/v1/models"}
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

    def test_invalid_job_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must contain"):
            queue_handler.normalize_job_input({})


class QueueProxyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        async def chat(request):
            body = await request.json()
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


if __name__ == "__main__":
    unittest.main()
