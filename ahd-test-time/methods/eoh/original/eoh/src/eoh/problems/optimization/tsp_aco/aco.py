import torch
from torch.distributions import Categorical


class ACO:
    def __init__(
        self,
        distances,
        heuristic,
        n_ants=30,
        decay=0.9,
        alpha=1,
        beta=1,
        device="cpu",
    ):
        self.problem_size = len(distances)
        self.distances = torch.tensor(distances, device=device) if not isinstance(distances, torch.Tensor) else distances
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

        self.pheromone = torch.ones_like(self.distances)
        self.heuristic = torch.tensor(heuristic, device=device) if not isinstance(heuristic, torch.Tensor) else heuristic

        self.shortest_path = None
        self.lowest_cost = float("inf")
        self.device = device

    @torch.no_grad()
    def run(self, n_iterations):
        for _ in range(n_iterations):
            paths = self.gen_path(require_prob=False)
            costs = self.gen_path_costs(paths)

            best_cost, best_idx = costs.min(dim=0)
            if best_cost < self.lowest_cost:
                self.shortest_path = paths[:, best_idx]
                self.lowest_cost = best_cost

            self.update_pheronome(paths, costs)

        return self.lowest_cost

    @torch.no_grad()
    def update_pheronome(self, paths, costs):
        self.pheromone = self.pheromone * self.decay
        for i in range(self.n_ants):
            path = paths[:, i]
            cost = costs[i]
            self.pheromone[path, torch.roll(path, shifts=1)] += 1.0 / cost
            self.pheromone[torch.roll(path, shifts=1), path] += 1.0 / cost

    @torch.no_grad()
    def gen_path_costs(self, paths):
        assert paths.shape == (self.problem_size, self.n_ants)
        u = paths.T
        v = torch.roll(u, shifts=1, dims=1)
        assert (self.distances[u, v] > 0).all()
        return torch.sum(self.distances[u, v], dim=1)

    def gen_path(self, require_prob=False):
        start = torch.randint(low=0, high=self.problem_size, size=(self.n_ants,), device=self.device)
        mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        mask[torch.arange(self.n_ants, device=self.device), start] = 0

        paths_list = [start]
        log_probs_list = []
        prev = start
        for _ in range(self.problem_size - 1):
            actions, log_probs = self.pick_move(prev, mask, require_prob)
            paths_list.append(actions)
            if require_prob:
                log_probs_list.append(log_probs)
                mask = mask.clone()
            prev = actions
            mask[torch.arange(self.n_ants, device=self.device), actions] = 0

        if require_prob:
            return torch.stack(paths_list), torch.stack(log_probs_list)
        return torch.stack(paths_list)

    def pick_move(self, prev, mask, require_prob):
        pheromone = self.pheromone[prev]
        heuristic = self.heuristic[prev]
        dist = (pheromone**self.alpha) * (heuristic**self.beta) * mask
        dist = Categorical(dist)
        actions = dist.sample()
        log_probs = dist.log_prob(actions) if require_prob else None
        return actions, log_probs
