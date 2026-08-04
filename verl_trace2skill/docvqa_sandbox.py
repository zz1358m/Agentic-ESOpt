from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from verl_trace2skill.docvqa_protocol import DOCVQA_IMAGE_PATH


@dataclass(frozen=True)
class SandboxResult:
    returncode: int
    text: str
    timed_out: bool = False


def _python_runtime_prefix() -> Path:
    # ``sys.executable`` is commonly a symlink to /usr/bin/python inside a
    # virtual environment. Resolving it loses the environment that contains
    # DocVQA's Pillow/OCR dependencies and leaves ``python`` unavailable in
    # the sandbox. ``sys.prefix`` is the runtime environment root by design.
    return Path(sys.prefix).absolute()


def _runtime_prefixes() -> list[Path]:
    prefixes = [_python_runtime_prefix()]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefixes.append(Path(conda_prefix))
    tool_prefix = os.environ.get("DOCVQA_TOOL_PREFIX")
    if tool_prefix:
        prefixes.append(Path(tool_prefix))
    return prefixes


def _existing_runtime_roots(extra: tuple[Path, ...]) -> list[Path]:
    roots = [Path("/usr"), Path("/bin"), Path("/lib"), Path("/lib64"), Path("/usr/local")]
    roots.extend(_runtime_prefixes())
    roots.extend(extra)
    result: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        resolved = str(root.resolve())
        if root.exists() and resolved not in seen:
            result.append(root)
            seen.add(resolved)
    return result


def _parent_dirs(path: Path) -> list[Path]:
    parents = []
    current = path.parent
    while current != Path("/"):
        parents.append(current)
        current = current.parent
    return list(reversed(parents))


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = "\n...[truncated]...\n"
    available = max(limit - len(marker), 2)
    left = available // 2
    right = available - left
    return text[:left] + marker + text[-right:]


def run_sandboxed_bash(
    command: str,
    *,
    image_path: str | Path | None = None,
    timeout: float = 20.0,
    max_output_chars: int = 6000,
    runtime_roots: tuple[str | Path, ...] = (),
    bwrap_path: str | Path | None = None,
) -> SandboxResult:
    image = Path(image_path).expanduser().resolve() if image_path is not None else None
    if image is not None and not image.is_file():
        raise FileNotFoundError(f"DocVQA image not found: {image}")
    bwrap = str(bwrap_path or shutil.which("bwrap") or "")
    if not bwrap:
        raise RuntimeError("bubblewrap executable 'bwrap' is required for DocVQA sandboxing")

    roots = _existing_runtime_roots(tuple(Path(path) for path in runtime_roots))
    path_parts = []
    runtime_prefixes = _runtime_prefixes()
    path_parts.extend(str(prefix / "bin") for prefix in runtime_prefixes if prefix.exists())
    path_parts.extend(["/usr/local/bin", "/usr/bin", "/bin"])
    sandbox_path = ":".join(dict.fromkeys(path_parts))
    args = [bwrap, "--unshare-all", "--new-session", "--die-with-parent", "--clearenv"]
    made_dirs: set[str] = set()
    for root in roots:
        for parent in _parent_dirs(root):
            value = str(parent)
            if value not in made_dirs:
                args.extend(["--dir", value])
                made_dirs.add(value)
        args.extend(["--ro-bind", str(root), str(root)])
    pip_prefixes = [*runtime_prefixes, Path("/usr"), Path("/usr/local")]
    for prefix in pip_prefixes:
        if not prefix.exists():
            continue
        pip_names = ("pip", "pip3", f"pip{os.sys.version_info.major}.{os.sys.version_info.minor}")
        for executable in pip_names:
            path = prefix / "bin" / executable
            if path.exists():
                args.extend(["--ro-bind", "/dev/null", str(path)])
        for pip_package in prefix.glob("lib/python*/site-packages/pip"):
            if pip_package.is_dir():
                args.extend(["--tmpfs", str(pip_package)])
        for pip_package in prefix.glob("lib/python*/dist-packages/pip"):
            if pip_package.is_dir():
                args.extend(["--tmpfs", str(pip_package)])
    args.extend(
        [
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--dir",
            "/workspace",
            "--chdir",
            "/workspace",
            "--setenv",
            "HOME",
            "/tmp",
            "--setenv",
            "TMPDIR",
            "/tmp",
            "--setenv",
            "PIP_NO_INDEX",
            "1",
            "--setenv",
            "PATH",
            sandbox_path,
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            command,
        ]
    )
    if image is not None:
        command_index = len(args) - 5
        args[command_index:command_index] = ["--ro-bind", str(image), DOCVQA_IMAGE_PATH]
    tool_prefix = os.environ.get("DOCVQA_TOOL_PREFIX")
    if tool_prefix:
        tool_root = Path(tool_prefix)
        library_dir = tool_root / "lib" / "x86_64-linux-gnu"
        tessdata_dir = tool_root / "share" / "tesseract-ocr" / "4.00" / "tessdata"
        if library_dir.is_dir():
            args[-5:-5] = ["--setenv", "LD_LIBRARY_PATH", str(library_dir)]
        if tessdata_dir.is_dir():
            args[-5:-5] = ["--setenv", "TESSDATA_PREFIX", str(tessdata_dir)]
    try:
        proc = subprocess.run(args, capture_output=True, timeout=timeout, check=False)
        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        text = stdout
        if stderr:
            text += ("\n[stderr]\n" if text else "[stderr]\n") + stderr
        if not text:
            text = f"Bash exited with code {proc.returncode} and no output."
        else:
            text = text.rstrip() + f"\n[exit_code] {proc.returncode}"
        return SandboxResult(proc.returncode, _truncate(text, max_output_chars))
    except subprocess.TimeoutExpired:
        return SandboxResult(124, f"Bash timed out after {timeout:.1f}s.", timed_out=True)
    except (OSError, ValueError) as exc:
        return SandboxResult(126, f"[ERROR] Failed to execute command: {exc}")
