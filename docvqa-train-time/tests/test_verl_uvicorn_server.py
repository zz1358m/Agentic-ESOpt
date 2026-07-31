from __future__ import annotations

import asyncio
import importlib.util
import json
import unittest
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI


SCRIPT = Path(__file__).resolve().parents[2] / "verl" / "verl" / "workers" / "rollout" / "utils.py"
SPEC = importlib.util.spec_from_file_location("verl_rollout_utils", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class VerlUvicornServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_returned_server_is_accepting_requests(self) -> None:
        app = FastAPI()

        @app.get("/health")
        async def health() -> dict[str, bool]:
            return {"ok": True}

        port, task = await MODULE.run_unvicorn(
            app,
            SimpleNamespace(),
            max_retries=1,
            lifespan="off",
        )
        try:
            response = await asyncio.to_thread(
                urllib.request.urlopen,
                f"http://127.0.0.1:{port}/health",
            )
            self.assertEqual(json.loads(response.read()), {"ok": True})
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task


if __name__ == "__main__":
    unittest.main()
