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
    if size > len(pop):
        size = len(pop)
    # Delete the worst individual
    pop_new = heapq.nsmallest(size, pop, key=lambda x: x['objective'])
    return pop_new
