#!/usr/bin/env python3
"""Install and verify the project-owned VAB-WebArena-Lite extensions."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXTENSIONS = ROOT / "webarena-train-time" / "vab_extensions"
PATCH = EXTENSIONS / "local_completion.patch"
OVERLAYS = {
    EXTENSIONS / "local_completion.py": Path("llms/providers/local_completion.py"),
    EXTENSIONS / "p_webrl_chat_qwen_action.json": Path(
        "agent/prompts/jsons/p_webrl_chat_qwen_action.json"
    ),
}
PATCH_MARKERS = {
    Path("llms/utils.py"): "from llms.providers.local_completion import",
    Path("llms/lm_config.py"): '"local_completion"',
    Path("llms/tokenizers.py"): '"local_completion"',
    Path("run.py"): 'parser.add_argument("--repetition_penalty"',
}


def same_file(left: Path, right: Path) -> bool:
    return left.is_file() and right.is_file() and left.read_bytes() == right.read_bytes()


def patch_state(vab_root: Path) -> str:
    states = []
    for relative_path, marker in PATCH_MARKERS.items():
        path = vab_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"VAB source file not found: {path}. Install the VAB-WebArena-Lite "
                "overlay from VisualAgentBench commit 9055fc2 first."
            )
        states.append(marker in path.read_text(encoding="utf-8"))
    if all(states):
        return "installed"
    if any(states):
        return "partial"
    return "missing"


def git_apply(vab_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "apply", *args, str(PATCH)],
        cwd=vab_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def overlay_conflicts(vab_root: Path) -> list[tuple[Path, Path]]:
    conflicts = []
    for source, relative_target in OVERLAYS.items():
        target = vab_root / relative_target
        if target.exists() and not same_file(source, target):
            conflicts.append((source, target))
    return conflicts


def check(vab_root: Path) -> tuple[bool, list[str]]:
    messages = []
    state = patch_state(vab_root)
    patch_ok = state == "installed"
    messages.append(f"VAB local_completion patch: {state}")
    overlays_ok = True
    for source, relative_target in OVERLAYS.items():
        target = vab_root / relative_target
        installed = same_file(source, target)
        overlays_ok &= installed
        messages.append(f"VAB overlay {relative_target}: {'installed' if installed else 'missing or different'}")
    return patch_ok and overlays_ok, messages


def install(vab_root: Path, force: bool) -> None:
    state = patch_state(vab_root)
    if state == "partial":
        raise RuntimeError(
            "The VAB local_completion patch is only partially installed. Restore the "
            "VAB files to VisualAgentBench commit 9055fc2 or complete the patch manually."
        )

    conflicts = overlay_conflicts(vab_root)
    if conflicts and not force:
        paths = ", ".join(str(target) for _, target in conflicts)
        raise RuntimeError(
            f"Refusing to overwrite different project-owned overlay file(s): {paths}. "
            "Inspect them first, then rerun with --force if replacement is intended."
        )

    if state == "missing":
        result = git_apply(vab_root, "--check")
        if result.returncode != 0:
            raise RuntimeError(
                "The VAB integration patch does not apply cleanly. Expected the "
                "VAB-WebArena-Lite overlay from VisualAgentBench commit 9055fc2.\n"
                + result.stdout.strip()
            )
        result = git_apply(vab_root)
        if result.returncode != 0:
            raise RuntimeError(result.stdout.strip())

    for source, relative_target in OVERLAYS.items():
        target = vab_root / relative_target
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main() -> None:
    default_vab = Path(os.environ.get("VAB_ROOT", ROOT / "data/webarena/vab-lite"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vab-root", type=Path, default=default_vab)
    parser.add_argument("--check", action="store_true", help="Verify without changing files.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace differing project-owned prompt/provider overlay files.",
    )
    args = parser.parse_args()
    vab_root = args.vab_root.expanduser().resolve()
    if not vab_root.is_dir():
        raise SystemExit(f"VAB root is not a directory: {vab_root}")

    if not args.check:
        install(vab_root, args.force)

    ready, messages = check(vab_root)
    for message in messages:
        print(message)
    if not ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
