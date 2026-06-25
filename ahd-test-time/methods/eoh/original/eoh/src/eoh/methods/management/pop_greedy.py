import heapq
import math


def _has_finite_objective(individual):
    objective = individual.get('objective')
    if objective is None:
        return False
    try:
        return math.isfinite(float(objective))
    except (TypeError, ValueError):
        return False

def population_management(pop,size):
    pop = [individual for individual in pop if _has_finite_objective(individual)]
    seen_objectives = set()
    unique_pop = []
    for individual in pop:
        objective = float(individual['objective'])
        if objective in seen_objectives:
            continue
        seen_objectives.add(objective)
        unique_pop.append(individual)
    pop = unique_pop
    if size > len(pop):
        size = len(pop)
    # Delete the worst individual
    pop_new = heapq.nsmallest(size, pop, key=lambda x: x['objective'])
    return pop_new
