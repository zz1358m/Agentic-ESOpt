from __future__ import annotations

import math
import re
from typing import Any


OPERATOR = re.compile(r"OP:\s*([a-z]\d)")
OBJECTIVE = re.compile(r"Obj:\s*(inf|[0-9.]+)")
FINISHED = re.compile(r"---\s+(\d+) of (\d+) populations finished")
POPULATION = re.compile(r"Pop Objs:\s*(.*)")


def parse_search_log(text: str) -> dict[str, list[dict[str, Any]]]:
    generations: list[dict[str, Any]] = []
    candidates: list[float] = []
    invalid = 0
    operator = "initial"
    pending_generation: int | None = None
    best_so_far = math.inf

    for line in text.splitlines():
        op = OPERATOR.search(line)
        if op:
            operator = op.group(1)

        if "Obj:" in line:
            for raw in OBJECTIVE.findall(line):
                if raw == "inf":
                    invalid += 1
                else:
                    candidates.append(float(raw))

        finished = FINISHED.search(line)
        if finished:
            pending_generation = int(finished.group(1))
            continue

        population = POPULATION.search(line)
        if population and pending_generation is not None:
            population_values = [float(value) for value in population.group(1).split()]
            best = min(population_values)
            best_so_far = min(best_so_far, best)
            generations.append(
                {
                    "generation": pending_generation,
                    "best": best,
                    "bestSoFar": best_so_far,
                    "candidates": candidates,
                    "invalidCandidates": invalid,
                    "operator": operator,
                }
            )
            candidates = []
            invalid = 0
            pending_generation = None

    return {"generations": generations}
