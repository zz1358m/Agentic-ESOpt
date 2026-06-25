from __future__ import annotations

import re
from pathlib import Path

import yaml


class SettingsPrompts:
    def __init__(self, problem_name: str):
        self.problem_name = str(problem_name)
        self.settings_root = self._resolve_settings_root()
        self.cfg = self._load_cfg()
        self.prompt_dir = self.settings_root / "prompts" / self.problem_name

        self.func_signature = (self.prompt_dir / "func_signature.txt").read_text(encoding="utf-8").format(version=2).strip()
        self.func_desc = (self.prompt_dir / "func_desc.txt").read_text(encoding="utf-8").strip()

        match = re.match(r"^def +(.+?)\((.*)\) *-> *(.*?) *:$", self.func_signature)
        if match is None:
            raise ValueError(f"Unsupported function signature: {self.func_signature!r}")

        self.prompt_func_name = match.group(1)
        inputs_raw = match.group(2).strip()
        self.prompt_func_inputs = []
        if inputs_raw:
            self.prompt_func_inputs = [txt.split(":")[0].strip() for txt in inputs_raw.split(",") if txt.strip()]

        if self.prompt_func_name.startswith("select_next_node"):
            self.prompt_func_outputs = ["next_node"]
        elif self.prompt_func_name.startswith("select_next_item"):
            self.prompt_func_outputs = ["next_item"]
        elif self.prompt_func_name.startswith("priority"):
            self.prompt_func_outputs = ["priority"]
        elif self.prompt_func_name.startswith("heuristics"):
            self.prompt_func_outputs = ["heuristics_matrix"]
        else:
            self.prompt_func_outputs = ["result"]

    @staticmethod
    def _resolve_settings_root() -> Path:
        for parent in Path(__file__).resolve().parents:
            candidate = parent / "data" / "ahd" / "settings"
            if candidate.is_dir():
                return candidate
        raise RuntimeError("Could not resolve data/ahd/settings root.")

    def _load_cfg(self) -> dict:
        cfg_path = self.settings_root / "cfg" / "problem" / f"{self.problem_name}.yaml"
        with cfg_path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def get_task(self):
        return self.cfg["description"]

    def get_func_name(self):
        return self.prompt_func_name

    def get_func_inputs(self):
        return self.prompt_func_inputs

    def get_func_outputs(self):
        return self.prompt_func_outputs

    def get_inout_inf(self):
        return self.func_desc

    def get_other_inf(self):
        return ""
