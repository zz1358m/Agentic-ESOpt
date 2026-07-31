import math
import re
from typing import Any


def _last_boxed_content(text: str) -> str | None:
    marker = r"\boxed{"
    start_marker = text.rfind(marker)
    if start_marker < 0:
        start_marker = text.rfind("\x08oxed{")
        if start_marker < 0:
            return None
        start = start_marker + len("\x08oxed{")
    else:
        start = start_marker + len(marker)
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1
    if depth == 0:
        return text[start : pos - 1].strip()
    return None


def _last_final_answer(text: str) -> str:
    matches = re.findall(
        r"^\s*(?:Final answer|Answer)\s*:\s*(.+?)\s*$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if matches:
        value = str(matches[-1]).strip()
        boxed = _last_boxed_content(value)
        return boxed if boxed is not None else value
    matches = re.findall(r"^\s*####\s*(.+?)\s*$", text, flags=re.MULTILINE)
    if matches:
        return str(matches[-1]).strip()
    boxed = _last_boxed_content(text)
    if boxed is not None:
        return boxed
    return ""


def _normalize_text(text: Any) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _anls(prediction: str, answers: list[str]) -> float:
    pred = _normalize_text(prediction)
    if not pred:
        return 0.0
    best = 0.0
    for answer in answers:
        gold = _normalize_text(answer)
        if not gold:
            continue
        dist = _levenshtein(pred, gold)
        denom = max(len(pred), len(gold))
        norm = dist / denom if denom else 1.0
        score = 1.0 - norm if norm < 0.5 else 0.0
        best = max(best, score)
    return float(best)


def _math_equal(prediction: str, ground_truth: Any) -> bool:
    pred = _last_final_answer(prediction)
    gold = str(ground_truth).strip()
    pred_norm = _normalize_text(pred).replace(" ", "")
    gold_norm = _normalize_text(gold).replace(" ", "")
    if pred_norm == gold_norm:
        return True
    try:
        if math.isclose(float(pred_norm), float(gold_norm), rel_tol=1e-9, abs_tol=1e-9):
            return True
    except Exception:
        pass

    try:
        from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

        extraction_config = [LatexExtractionConfig(), ExprExtractionConfig()]
        parsed_pred = parse(pred, extraction_config=extraction_config)
        parsed_gold = parse(gold, extraction_config=extraction_config)
        if parsed_pred and parsed_gold and verify(parsed_gold, parsed_pred):
            return True
    except Exception:
        pass
    return False


def _used_bash_action(text: str) -> bool:
    if re.search(r"Action:\s*\{.*?\"name\"\s*:\s*\"bash\"", text, flags=re.IGNORECASE | re.DOTALL):
        return True
    if re.search(r"<tool_call>\s*\{.*?\"name\"\s*:\s*\"bash\"", text, flags=re.IGNORECASE | re.DOTALL):
        return True
    return "Observation from bash:" in text


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, float]:
    task = (data_source or "").lower()
    tool_used = (
        "Observation from bash:" in solution_str
        if "docvqa" in task
        else _used_bash_action(solution_str)
    )
    if not tool_used:
        if "docvqa" in task:
            return {"score": 0.0, "acc": 0.0, "anls": 0.0, "tool_used": 0.0}
        return {"score": 0.0, "acc": 0.0, "tool_used": 0.0}

    if "docvqa" in task:
        answers = ground_truth if isinstance(ground_truth, list) else [ground_truth]
        pred = _last_final_answer(solution_str)
        anls = _anls(pred, [str(x) for x in answers])
        acc = float(anls > 0.5)
        return {"score": anls, "acc": acc, "anls": anls, "tool_used": 1.0}

    score = float(_math_equal(solution_str, ground_truth))
    return {"score": score, "acc": score, "tool_used": 1.0}
