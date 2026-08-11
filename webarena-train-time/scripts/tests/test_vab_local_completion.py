from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "vab_extensions" / "local_completion.py"
)
SPEC = importlib.util.spec_from_file_location("vab_local_completion", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class LocalCompletionTest(unittest.TestCase):
    def test_chat_maps_completion_url_and_decodes_openai_response(self):
        response = FakeResponse(
            {"choices": [{"message": {"role": "assistant", "content": "do(action=\"Scroll Down\")"}}]}
        )
        with mock.patch.object(ADAPTER.requests, "post", return_value=response) as post:
            with mock.patch.dict(os.environ, {"WEBRL_LOCAL_ENABLE_THINKING": "false"}):
                result = ADAPTER.generate_from_local_chat_completion(
                    messages=[{"role": "user", "content": "task"}],
                    endpoint="http://127.0.0.1:11013/completions",
                    model="Qwen3.5-27B",
                    temperature=0.7,
                    max_tokens=2048,
                    top_p=0.8,
                    top_k=20,
                    min_p=0.0,
                    presence_penalty=1.5,
                    repetition_penalty=1.0,
                )

        self.assertEqual(result, 'do(action="Scroll Down")')
        self.assertEqual(post.call_args.args[0], "http://127.0.0.1:11013/v1/chat/completions")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["params"]["top_k"], 20)
        self.assertEqual(payload["params"]["presence_penalty"], 1.5)
        self.assertEqual(payload["params"]["enable_thinking"], "false")

    def test_completion_uses_raw_prompt_endpoint_and_decodes_content(self):
        response = FakeResponse({"content": ["exit(message=\"done\")"]})
        with mock.patch.object(ADAPTER.requests, "post", return_value=response) as post:
            result = ADAPTER.generate_from_local_completion(
                prompt="prompt",
                endpoint="http://127.0.0.1:11013/completions",
                temperature=0.0,
                max_tokens=128,
                top_p=1.0,
                stop_token="<stop>",
            )

        self.assertEqual(result, 'exit(message="done")')
        self.assertEqual(post.call_args.args[0], "http://127.0.0.1:11013/completions")
        params = post.call_args.kwargs["json"]["params"]
        self.assertFalse(params["do_sample"])
        self.assertFalse(params["use_chat_template"])
        self.assertEqual(params["stop"], ["<stop>"])


if __name__ == "__main__":
    unittest.main()
