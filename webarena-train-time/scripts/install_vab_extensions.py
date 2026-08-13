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
JUDGE_PATCH = EXTENSIONS / "evaluation_judge.patch"
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
JUDGE_MODEL_EXPRESSION = 'model="gpt-4.1-mini"'
LEGACY_CONFIGURABLE_JUDGE_EXPRESSION = (
    'model=os.environ.get("WEBRL_EVAL_MODEL", "gpt-4.1-mini")'
)


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


def git_apply(vab_root: Path, patch: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "apply", *args, str(patch)],
        cwd=vab_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def judge_patch_state(vab_root: Path) -> str:
    helper = vab_root / "evaluation_harness/helper_functions.py"
    if not helper.is_file():
        raise FileNotFoundError(f"VAB evaluator source file not found: {helper}")
    text = helper.read_text(encoding="utf-8")
    model_sites = text.count(JUDGE_MODEL_EXPRESSION)
    configurable_sites = text.count(LEGACY_CONFIGURABLE_JUDGE_EXPRESSION)
    if model_sites == 2:
        return "installed"
    if configurable_sites == 2 and model_sites == 0:
        return "legacy-configurable"
    if model_sites == 0 and configurable_sites == 0:
        return "missing"
    return "partial"


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
    judge_state = judge_patch_state(vab_root)
    judge_ok = judge_state == "installed"
    messages.append(
        "VAB GPT judge patch: "
        f"{judge_state} (hard-coded gpt-4.1-mini)"
    )
    overlays_ok = True
    for source, relative_target in OVERLAYS.items():
        target = vab_root / relative_target
        installed = same_file(source, target)
        overlays_ok &= installed
        messages.append(f"VAB overlay {relative_target}: {'installed' if installed else 'missing or different'}")
    return patch_ok and judge_ok and overlays_ok, messages


def install(vab_root: Path, force: bool) -> None:
    state = patch_state(vab_root)
    if state == "partial":
        raise RuntimeError(
            "The VAB local_completion patch is only partially installed. Restore the "
            "VAB files to VisualAgentBench commit 9055fc2 or complete the patch manually."
        )

    judge_state = judge_patch_state(vab_root)
    if judge_state == "partial":
        raise RuntimeError(
            "The VAB GPT judge patch is only partially installed. Restore "
            "evaluation_harness/helper_functions.py to the VAB overlay from "
            "VisualAgentBench commit 9055fc2, then rerun this installer."
        )

    if judge_state == "legacy-configurable":
        helper = vab_root / "evaluation_harness/helper_functions.py"
        text = helper.read_text(encoding="utf-8").replace(
            LEGACY_CONFIGURABLE_JUDGE_EXPRESSION,
            JUDGE_MODEL_EXPRESSION,
        )
        if "os." not in text:
            text = text.replace("import os\n", "", 1)
        helper.write_text(text, encoding="utf-8")
        judge_state = "installed"

    conflicts = overlay_conflicts(vab_root)
    if conflicts and not force:
        paths = ", ".join(str(target) for _, target in conflicts)
        raise RuntimeError(
            f"Refusing to overwrite different project-owned overlay file(s): {paths}. "
            "Inspect them first, then rerun with --force if replacement is intended."
        )

    if state == "missing":
        result = git_apply(vab_root, PATCH, "--check")
        if result.returncode != 0:
            raise RuntimeError(
                "The VAB integration patch does not apply cleanly. Expected the "
                "VAB-WebArena-Lite overlay from VisualAgentBench commit 9055fc2.\n"
                + result.stdout.strip()
            )
        result = git_apply(vab_root, PATCH)
        if result.returncode != 0:
            raise RuntimeError(result.stdout.strip())

    if judge_state == "missing":
        result = git_apply(vab_root, JUDGE_PATCH, "--check")
        if result.returncode != 0:
            raise RuntimeError(
                "The VAB GPT judge patch does not apply cleanly. Expected the "
                "VAB-WebArena-Lite overlay from VisualAgentBench commit 9055fc2.\n"
                + result.stdout.strip()
            )
        result = git_apply(vab_root, JUDGE_PATCH)
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
