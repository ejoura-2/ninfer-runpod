import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).parents[1]
ENTRYPOINT_SPEC = importlib.util.spec_from_file_location("entrypoint", ROOT / "entrypoint.py")
entrypoint = importlib.util.module_from_spec(ENTRYPOINT_SPEC)
assert ENTRYPOINT_SPEC.loader is not None
ENTRYPOINT_SPEC.loader.exec_module(entrypoint)
sys.modules["entrypoint"] = entrypoint

SPEC = importlib.util.spec_from_file_location(
    "queue_entrypoint", ROOT / "queue_entrypoint.py"
)
queue_entrypoint = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(queue_entrypoint)


class QueueEntrypointTests(unittest.TestCase):
    def test_wait_for_ninfer_returns_when_healthy(self):
        process = Mock()
        process.poll.return_value = None
        with patch.object(queue_entrypoint.entrypoint, "upstream_ready", return_value=True):
            queue_entrypoint.wait_for_ninfer(process, 8080, 10)

    def test_wait_for_ninfer_fails_when_process_exits(self):
        process = Mock()
        process.poll.return_value = 2
        process.returncode = 2
        with self.assertRaisesRegex(RuntimeError, "exited during startup"):
            queue_entrypoint.wait_for_ninfer(process, 8080, 10)


if __name__ == "__main__":
    unittest.main()
