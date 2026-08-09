# source: /home/zhi/Agentic-ESOpt/cache/active_runs/aco_tsp_train_sample_t1000_aco_tsp_sample_t1000_rep1_20260718_091608/results/pops_best/population_generation_50.json
# method: sample, prefix=1000, batch_size=20
# task: aco_tsp, rep: 1
# train_objective: 5.90618

# numerical_stabilization: exp_log_upper_clip_60

import random
import numpy as np
from scipy.special import xlogy

def heuristics(distance_matrix):
    n = len(distance_matrix)
    edge_log_prob = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i!= j:
                edge_log_prob[i, j] = 1 / distance_matrix[i, j]
    for _ in range(1000):
        visited = list(range(n))
        random.shuffle(visited)
        path = [visited[0]]
        while len(path) < n:
            min_dist = float('inf')
            argmin_dist = None
            for i in visited:
                if i not in path:
                    dist = distance_matrix[path[-1], i]
                    if dist < min_dist:
                        min_dist = dist
                        argmin_dist = i
            path.append(argmin_dist)
        path.append(path[0])
        for i in range(n):
            for j in range(n):
                if i!= j and distance_matrix[path[i], path[j]] == 1:
                    edge_log_prob[i, j] += 1
    stable_log_prob = np.minimum(edge_log_prob, 60.0)
    return np.exp(stable_log_prob)

def heuristics_v2(distance_matrix):
    edge_log_prob = heuristics(distance_matrix)
    return np.exp(edge_log_prob)
