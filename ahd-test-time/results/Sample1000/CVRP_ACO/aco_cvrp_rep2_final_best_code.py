# source: /home/zhi/Dynamic-Agent/cache/active_runs/aco_cvrp_train_sample_t1000_aco_cvrp_sample_t1000_rep2_20260718_144918/results/pops_best/population_generation_50.json
# method: sample, prefix=1000, batch_size=20
# task: aco_cvrp, rep: 2
# train_objective: 10.22879

import numpy as np
import random

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = distance_matrix.shape[0]
    heuristics_matrix = np.zeros(distance_matrix.shape)

    for u in range(n):
        visited_nodes = [0]
        visited_demands = [demands[0]]
        total_demand = 0
        
        while len(visited_nodes) < n:
            promising_edges = []
            for v in range(n):
                if v not in visited_nodes and total_demand + demands[v] <= capacity:
                    heuristics_matrix[u, v] = 1 / distance_matrix[u, v]
                    promising_edges.append(v)
            if not promising_edges:
                break
            next_node = random.choice(promising_edges)
            visited_nodes.append(next_node)
            visited_demands.append(demands[next_node])
            total_demand += demands[next_node]
            
    return heuristics_matrix
