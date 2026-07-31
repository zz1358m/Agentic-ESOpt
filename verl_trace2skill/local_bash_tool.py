import asyncio
import os
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

from verl_trace2skill.docvqa_sandbox import run_sandboxed_bash


class LocalBashTool(BaseTool):
    """A small per-trajectory shell tool for verl multi-turn rollout."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instances: dict[str, dict[str, Any]] = {}
        self._default_cwd = str(Path(config.get("default_cwd", os.getcwd())).resolve())
        self._timeout = float(config.get("timeout", 20))
        self._max_output_chars = int(config.get("max_output_chars", 6000))
        self._sandbox = bool(config.get("sandbox", False))

    async def create(
        self,
        instance_id: Optional[str] = None,
        cwd: Optional[str] = None,
        image_path: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        **kwargs,
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        create_kwargs = kwargs.get("create_kwargs", {})
        cwd = cwd or create_kwargs.get("cwd") or self._default_cwd
        image_path = image_path or create_kwargs.get("image_path")
        env = env or create_kwargs.get("env") or {}
        self._instances[instance_id] = {
            "cwd": str(Path(cwd).resolve()),
            "env": {str(k): str(v) for k, v in env.items()},
            "image_path": str(Path(image_path).resolve()) if image_path else None,
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        command = parameters.get("command")
        if command is None:
            command = parameters.get("cmd")
        if command is None:
            command = parameters.get("code")
        if command is None:
            return ToolResponse(text="[ERROR] Missing required field: command"), 0.0, {"bash_error": 1.0}
        if not isinstance(command, str):
            command = str(command)

        state = self._instances.get(instance_id, {})
        if self._sandbox:
            image_path = state.get("image_path")
            if not image_path:
                return ToolResponse(text="[ERROR] Missing sandbox image_path"), 0.0, {"bash_error": 1.0}
            result = await asyncio.to_thread(
                run_sandboxed_bash,
                command,
                image_path=image_path,
                timeout=float(kwargs.get("timeout", self._timeout)),
                max_output_chars=self._max_output_chars,
            )
            metrics = {"bash_returncode": float(result.returncode)}
            if result.timed_out:
                metrics.update({"bash_error": 1.0, "bash_timeout": 1.0})
            return ToolResponse(text=f"Observation from bash:\n{result.text}"), 0.0, metrics

        cwd = kwargs.get("cwd") or state.get("cwd") or self._default_cwd
        timeout = float(kwargs.get("timeout", self._timeout))
        env = os.environ.copy()
        env.update(state.get("env", {}))
        env.update({str(k): str(v) for k, v in kwargs.get("env", {}).items()})

        try:
            proc = await asyncio.create_subprocess_exec(
                "bash",
                "-lc",
                command,
                cwd=cwd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            text = self._format_output(proc.returncode, stdout, stderr)
            metrics = {"bash_returncode": float(proc.returncode)}
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.communicate()
            except Exception:
                pass
            text = f"Bash timed out after {timeout:.1f}s."
            metrics = {"bash_error": 1.0, "bash_timeout": 1.0}
        except Exception as exc:
            text = f"[ERROR] Failed to execute command: {exc}"
            metrics = {"bash_error": 1.0}

        if len(text) > self._max_output_chars:
            half = max(1, (self._max_output_chars - 80) // 2)
            text = text[:half] + "\n...[truncated]...\n" + text[-half:]
        return ToolResponse(text=f"Observation from bash:\n{text}"), 0.0, metrics

    def _format_output(self, returncode: int, stdout: bytes, stderr: bytes) -> str:
        out = stdout.decode("utf-8", errors="replace")
        err = stderr.decode("utf-8", errors="replace")
        output = ""
        if out:
            output += out
        if err:
            output += ("\n[stderr]\n" if output else "[stderr]\n") + err
        if not output:
            return f"Bash exited with code {returncode} and no output."
        return output.rstrip() + f"\n[exit_code] {returncode}"

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)
