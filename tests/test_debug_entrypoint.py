import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "debug_entrypoint", Path(__file__).parents[1] / "debug_entrypoint.py"
)
debug_entrypoint = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(debug_entrypoint)


class DebugEntrypointTests(unittest.TestCase):
    def test_public_key_is_required(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(SystemExit, "PUBLIC_KEY is required"):
                debug_entrypoint.configure_ssh_key()

    def test_public_key_is_written_with_secure_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            ssh_dir = Path(directory) / ".ssh"
            real_path = debug_entrypoint.pathlib.Path

            def mapped_path(value):
                if value == "/root/.ssh":
                    return ssh_dir
                return real_path(value)

            with patch.dict(os.environ, {"PUBLIC_KEY": "ssh-ed25519 test-key"}, clear=True):
                with patch.object(debug_entrypoint.pathlib, "Path", side_effect=mapped_path):
                    debug_entrypoint.configure_ssh_key()

            authorized_keys = ssh_dir / "authorized_keys"
            self.assertEqual(authorized_keys.read_text(), "ssh-ed25519 test-key\n")
            if os.name != "nt":
                self.assertEqual(authorized_keys.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
