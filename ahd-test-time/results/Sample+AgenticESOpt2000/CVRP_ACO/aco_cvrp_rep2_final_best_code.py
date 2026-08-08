# source: /home/zhi/Dynamic-Agent/cache/active_runs/aco_cvrp_train_sample_es_pop20_gen100_sigma0.001_alpha0.0005_aco_cvrp_sample_es_current_cosine_t2000_rep2_queue_b_gpu4_7_20260720_030717/results/pops_best/population_generation_100.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=100, samples=2000, sigma=0.001->0, alpha=0.0005
# task: aco_cvrp, rep: 2
# train_objective: 9.57838

import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = len(demands)
    nodes = np.arange(n)
    node_pairs = [(i, j) for i in nodes for j in nodes if i < j]
    probabilities = np.zeros((n, n))
    
    for i, j in node_pairs:
        if i == 0 or j == 0:
            continue
        total_demand = demands[i] + demands[j]
        if total_demand <= capacity:
            probability = 1 / ((distance_matrix[i, j] / (capacity - total_demand + 1e-6))**2)
        else:
            probability = 0
        probabilities[i, j] = probability
        probabilities[j, i] = probability
    
    return probabilities

def heuristics_v2(distance_matrix, coordinates, demands, capacity, num_samples=1000):
    n = len(demands)
    probabilities = heuristics(distance_matrix, coordinates, demands, capacity)
    sample_paths = [np.random.permutation(np.arange(n)) for _ in range(num_samples)]
    
    sampled_heuristics_matrices = []
    for _ in range(num_samples):
        path = sample_paths[np.random.randint(0, len(sample_paths))]
        sampled_heuristics = np.zeros((n, n))
        for i in range(n):
            sampled_heuristics[path[i], path[(i+1)%n]] = probabilities[path[i], path[(i+1)%n]]
            sampled_heuristics[path[(i+1)%n], path[i]] = probabilities[path[(i+1)%n], path[i]]
        sampled_heuristics_matrices.append(sampled_heuristics)
    
    best_sample = min(sampled_heuristics_matrices, key=lambda x: np.sum(x[0, :]))
    return best_sample
