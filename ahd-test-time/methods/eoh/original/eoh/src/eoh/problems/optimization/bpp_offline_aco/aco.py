from math import floor
from typing import Annotated, List, Tuple

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.int_]
FloatArray = npt.NDArray[np.float64]


def organize_path(path: IntArray) -> Tuple[int, IntArray]:
    order = {}
    result = np.zeros_like(path)
    for i, v in enumerate(path):
        if v in order:
            result[i] = order[v]
        else:
            result[i] = order[v] = len(order)
    return len(order), result


def calculate_path_fitness(vacancies: List[int], capacity: int) -> float:
    occupied = capacity - np.array(vacancies, dtype=float)
    return ((occupied / capacity) ** 2).sum().item() / len(vacancies)


def greedy_sample(prob: FloatArray) -> int:
    return prob.argmax().item()


def random_sample_discrete_distribution(prob: FloatArray) -> int:
    cumprob = np.cumsum(prob)
    sampled = np.searchsorted(cumprob, next(uniform_generator) * cumprob[-1]).item()
    return sampled if sampled < len(cumprob) else len(cumprob) - 1


def uniform_number_generator(batch_size=500):
    while 1:
        numbers = np.random.random(batch_size)
        for n in numbers:
            yield n.item()


uniform_generator = uniform_number_generator()


class ACO(object):
    def __init__(
        self,
        demand: IntArray,
        heuristic: FloatArray,
        capacity: int,
        n_ants=20,
        decay=0.95,
        alpha=1,
        beta=1,
        greedy=False,
    ):
        self.problem_size = len(demand)
        self.capacity = capacity
        self.demand = demand
        assert self.demand.max() <= self.capacity

        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

        self.pheromone: FloatArray = np.ones((self.problem_size, self.problem_size))
        heuristic[heuristic > 1e6] = 1e6
        heuristic[heuristic < 1e-6] = 1e-6
        heuristic = heuristic / heuristic.max()
        heuristic[heuristic < 1e-6] = 1e-6
        self.heuristic: FloatArray = heuristic

        self.shortest_path: IntArray = np.arange(self.problem_size)
        self.best_cost = self.problem_size
        self._ordinal: IntArray = np.arange(self.problem_size, dtype=int)
        self.greedy_mode = greedy

    def run(self, iterations: int) -> Tuple[int, IntArray]:
        for _ in range(iterations):
            prob = self.pheromone**self.alpha * self.heuristic**self.beta
            paths, costs, fitnesses = self.gen_paths(self.n_ants, prob)
            best_index = costs.argmin()
            best_cost = costs[best_index].item()
            if best_cost < self.best_cost:
                self.shortest_path = paths[best_index]
                self.best_cost = best_cost
            self.update_pheronome(paths, fitnesses)
        return organize_path(self.shortest_path)

    def sample_only(self, count: int) -> Tuple[int, IntArray]:
        self.greedy_mode = True
        paths, costs, _ = self.gen_paths(count, self.heuristic)
        best_index = costs.argmin()
        best_path = paths[best_index]
        return organize_path(best_path)

    def update_pheronome(self, paths: List[IntArray], fitnesses: FloatArray):
        delta_phe = np.zeros_like(self.pheromone)
        for path, f in zip(paths, fitnesses):
            delta_phe[path[:, None] == path[None, :]] += f / self.n_ants
        self.pheromone *= self.decay
        self.pheromone += delta_phe

    def gen_paths(self, count: int, prob: FloatArray):
        paths, costs, fitnesses = [], [], []
        for _ in range(count):
            path, cost, fitness = self.sample_path(prob)
            paths.append(path)
            costs.append(cost)
            fitnesses.append(fitness)
        return paths, np.array(costs, dtype=int), np.array(fitnesses, dtype=float)

    def sample_path(
        self, prob: FloatArray
    ) -> Tuple[
        Annotated[IntArray, "sampled path"],
        Annotated[int, "used bins"],
        Annotated[float, "fitness"],
    ]:
        sample_func = greedy_sample if self.greedy_mode else random_sample_discrete_distribution

        path = np.ones(self.problem_size, dtype=int) * -1
        valid_items = np.ones(self.problem_size, dtype=bool)
        current_bin = item_count = 0
        vacancies = []
        bin_vacancy = self.capacity
        bin_items = np.zeros_like(valid_items)

        for _ in range(self.problem_size):
            mask = np.bitwise_and(self.demand <= bin_vacancy, valid_items)
            if not np.any(mask):
                vacancies.append(bin_vacancy)
                bin_vacancy, item_count = self.capacity, 0
                current_bin += 1
                bin_items[:] = False
                selected = self.random_select(valid_items)
            else:
                if item_count == 0:
                    selected = self.random_select(mask)
                else:
                    item_prob = (prob[bin_items].sum(0) / item_count + 1e-5) * mask
                    selected = sample_func(item_prob)

            bin_items[selected] = True
            bin_vacancy -= self.demand[selected]
            valid_items[selected] = False
            path[selected] = current_bin
            item_count += 1

        vacancies.append(bin_vacancy)
        fitness = calculate_path_fitness(vacancies, self.capacity)
        return path, len(vacancies), fitness

    def random_select(self, mask: npt.NDArray[np.bool_]) -> int:
        valid = self._ordinal[mask]
        return valid[floor(next(uniform_generator) * len(valid))].item()
