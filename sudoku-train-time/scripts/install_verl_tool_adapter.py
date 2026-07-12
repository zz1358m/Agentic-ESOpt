#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"{src} -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verl-tool-root", required=True, help="Path to a verl-tool checkout.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    verl_root = Path(args.verl_tool_root).resolve()
    src_root = repo_root / "sudoku-train-time" / "verl_tool"
    copy_file(src_root / "sudoku.py", verl_root / "verl_tool" / "servers" / "tools" / "sudoku.py")
    copy_file(
        src_root / "sudoku_binary.py",
        verl_root / "verl_tool" / "workers" / "reward_manager" / "sudoku_binary.py",
    )
    print("Set DYNAMIC_AGENT_ROOT to this repository root before starting verl-tool.")
    print(f"PowerShell: $env:DYNAMIC_AGENT_ROOT='{repo_root}'")
    print(f"Bash: export DYNAMIC_AGENT_ROOT='{repo_root.as_posix()}'")


if __name__ == "__main__":
    main()
