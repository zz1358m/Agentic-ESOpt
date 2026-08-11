from __future__ import annotations

import json
import importlib.util
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from urllib import request


ROOT = Path(__file__).resolve().parents[2]
SERVER = (
    ROOT
    / "ahd-test-time/methods/eoh/original/eoh/src/eoh/llm_local_server/llama31_instruct_server.py"
)
BATCH_RETRY = SERVER.with_name("batch_retry.py")


def load_batch_retry():
    spec = importlib.util.spec_from_file_location("llm_server_batch_retry", BATCH_RETRY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {BATCH_RETRY}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_port(port: int, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, _ = process.communicate()
            raise RuntimeError(f"server exited with {process.returncode}:\n{stdout}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError(f"server did not open port {port}")


def save_tiny_model(path: Path) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import AutoModelForCausalLM, GPT2Config, PreTrainedTokenizerFast

    backend = Tokenizer(
        WordLevel(
            {"<pad>": 0, "<eos>": 1, "<unk>": 2, "alpha": 3, "beta": 4},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(path)
    model = AutoModelForCausalLM.from_config(
        GPT2Config(
            vocab_size=5,
            n_positions=32,
            n_embd=16,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=0,
        )
    )
    with __import__("torch").no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for module in model.modules():
            if module.__class__.__name__ == "LayerNorm":
                module.weight.fill_(1.0)
                module.bias.zero_()
        model.transformer.wte.weight[3, 0] = 10.0
        model.transformer.wte.weight[4, 1] = 10.0
    model.save_pretrained(path)


class TransformersServerBatchContractTests(unittest.TestCase):
    def test_oom_splitting_preserves_input_output_order(self) -> None:
        batch_retry = load_batch_retry()
        calls = []

        class FakeOutOfMemoryError(RuntimeError):
            pass

        def run_batch(items):
            calls.append(list(items))
            if len(items) > 1:
                raise FakeOutOfMemoryError()
            return [f"output:{items[0]}"]

        parts = batch_retry.run_with_oom_splitting(
            ["alpha", "beta", "gamma"],
            run_batch=run_batch,
            is_oom=lambda exc: isinstance(exc, FakeOutOfMemoryError),
        )

        self.assertEqual(
            [value for part in parts for value in part],
            ["output:alpha", "output:beta", "output:gamma"],
        )
        self.assertEqual(calls[0], ["alpha", "beta", "gamma"])

    def test_completions_returns_one_result_per_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "tiny-model"
            model_path.mkdir()
            save_tiny_model(model_path)
            port = free_port()
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")])
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(SERVER),
                    "--path",
                    str(model_path),
                    "--d",
                    "cpu",
                    "--dtype",
                    "auto",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--seed",
                    "20260811",
                ],
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                wait_for_port(port, process)
                payload = json.dumps(
                    {
                        "prompt": ["alpha", "alpha beta"],
                        "max_tokens": 1,
                        "temperature": 0.0,
                        "use_chat_template": False,
                    }
                ).encode()
                req = request.Request(
                    f"http://127.0.0.1:{port}/completions",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with request.urlopen(req, timeout=30) as response:
                    result = json.loads(response.read().decode())

                self.assertEqual(result["content"], ["alpha", "beta"])
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
                if process.stdout is not None:
                    process.stdout.close()


if __name__ == "__main__":
    unittest.main()
