# source: /home/zhi/Agentic-ESOpt/cache/active_runs/aco_bpp_train_sample_es_pop20_gen100_sigma0.001_alpha0.0005_aco_bpp_sample_es_current_cosine_t2000_rep1_queue_b_gpu4_7_20260720_030717/results/pops_best/population_generation_100.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=100, samples=2000, sigma=0.001->0, alpha=0.0005
# task: aco_bpp, rep: 1
# train_objective: 202.4

import numpy as np

def heuristics_v2(demand, capacity):
    n = len(demand)
    demand = np.array(demand)
    heuristics = np.zeros((n, n))
    indices_bins = [[] for _ in range(n)]
    for i in np.argsort(-demand):
        new_bin = True
        for j, bi in enumerate(indices_bins):
            if demand[i] <= capacity - sum(demand[bi]):
                indices_bins[j].append(i)
                break
        else:
            indices_bins.append([i])
            new_bin = False
    for i, bi in enumerate(indices_bins):
        heuristics[np.ix_(bi, bi)] = np.where(np.in1d(bi, bi), 1, 0)
    return heuristics

# Example usage
demand = [5, 3, 4, 2, 1]
capacity = 5
heuristics = heuristics_v2(demand, capacity)
print(heuristics)
