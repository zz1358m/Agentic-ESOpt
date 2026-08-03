import asyncio

import pytest
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager, _await_remote_result
from verl.workers.rollout.replica import TokenOutput


class _RemoteMethod:
    def __init__(self, function):
        self.function = function

    def remote(self, **kwargs):
        return self.function(**kwargs)


class _FakeServer:
    def __init__(self):
        self.generate_calls = 0
        self.aborted = []
        self.generate = _RemoteMethod(self._generate)
        self.abort_request = _RemoteMethod(self._abort)

    async def _generate(self, **kwargs):
        self.generate_calls += 1
        if self.generate_calls == 1:
            await asyncio.sleep(0.05)
        return TokenOutput(token_ids=[7, 8], log_probs=None)

    async def _abort(self, **kwargs):
        self.aborted.append(kwargs)


def test_generate_timeout_aborts_and_retries(monkeypatch):
    monkeypatch.setenv("TRACE2SKILL_GENERATE_TIMEOUT_SECONDS", "0.01")
    monkeypatch.setenv("TRACE2SKILL_GENERATE_MAX_ATTEMPTS", "2")
    server = _FakeServer()
    manager = AsyncLLMServerManager(OmegaConf.create({}), [server])

    output = asyncio.run(
        manager.generate(
            "trajectory-1",
            prompt_ids=[1, 2],
            sampling_params={"max_new_tokens": 3},
        )
    )

    assert output.token_ids == [7, 8]
    assert server.generate_calls == 2
    assert server.aborted == [{"request_id": "trajectory-1"}]


def test_generate_timeout_is_bounded(monkeypatch):
    monkeypatch.setenv("TRACE2SKILL_GENERATE_TIMEOUT_SECONDS", "0.001")
    monkeypatch.setenv("TRACE2SKILL_GENERATE_MAX_ATTEMPTS", "1")
    server = _FakeServer()
    manager = AsyncLLMServerManager(OmegaConf.create({}), [server])

    with pytest.raises(TimeoutError, match="trajectory-1"):
        asyncio.run(
            manager.generate(
                "trajectory-1",
                prompt_ids=[1, 2],
                sampling_params={"max_new_tokens": 3},
            )
        )

    assert server.aborted == [{"request_id": "trajectory-1"}]


def test_remote_result_helper_retries_a_lost_reward_reply():
    calls = 0

    async def pending():
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(0.05)
        return {"reward_score": 1.0}

    result = asyncio.run(
        _await_remote_result(
            pending,
            timeout_seconds=0.01,
            max_attempts=2,
            label="reward trajectory-1",
        )
    )

    assert result == {"reward_score": 1.0}
    assert calls == 2
