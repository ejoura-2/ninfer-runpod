import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "entrypoint", Path(__file__).parents[1] / "entrypoint.py"
)
entrypoint = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(entrypoint)


class EntrypointTests(unittest.TestCase):
    def test_explicit_model_path(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.ninfer"
            model.touch()
            with patch.dict(os.environ, {"MODEL_PATH": str(model)}, clear=True):
                self.assertEqual(entrypoint.resolve_model_path(), model)

    def test_command_enables_full_context_vision_and_thinking(self):
        with patch.dict(os.environ, {}, clear=True):
            command = entrypoint.build_command(Path("/models/model.ninfer"))
        self.assertIn("262144", command)
        self.assertIn("--vision", command)
        self.assertIn("mtp", command)
        self.assertIn("--preserve-thinking", command)
        self.assertIn("int8", command)
        self.assertEqual(command[command.index("--port") + 1], "8080")

    def test_ninfer_uses_public_port(self):
        with patch.dict(os.environ, {"PORT": "9000"}, clear=True):
            command = entrypoint.build_command(Path("/models/model.ninfer"))
        self.assertEqual(command[command.index("--port") + 1], "9000")

    def test_optional_ninfer_key_is_forwarded(self):
        with patch.dict(os.environ, {"NINFER_API_KEY": "secret"}, clear=True):
            command = entrypoint.build_command(Path("/models/model.ninfer"))
        self.assertEqual(command[-2:], ["--api-key", "secret"])


if __name__ == "__main__":
    unittest.main()
