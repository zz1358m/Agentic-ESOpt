# source: /home/zhi/Agentic-ESOpt/cache/active_runs/aco_tsp_train_sample_es_pop20_gen100_sigma0.001_alpha0.0005_aco_tsp_sample_es_current_cosine_t2000_rep1_queue_a_gpu0_3_20260720_030717/results/pops_best/population_generation_100.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=100, samples=2000, sigma=0.001->0, alpha=0.0005
# task: aco_tsp, rep: 1
# train_objective: 5.87114

import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations

def heuristics_v2(distance_matrix):
    num_nodes = len(distance_matrix)
    heuristics_matrix = np.zeros((num_nodes, num_nodes))
    
    for i, j in combinations(range(num_nodes), 2):
        if i == j:
            heuristics_matrix[i, j] = 0
            heuristics_matrix[j, i] = 0
            continue
        
        edge_value = (1 / distance_matrix[i, j]) * (1 / distance_matrix[i, j]) * (1 / distance_matrix[i, j]) 
        heuristics_matrix[i, j] = edge_value
        
        if i < j:
            heuristics_matrix[j, i] = edge_value
    
    # Optional: Perform sampling to reduce precision but increase speed
    return heuristics_matrix

# Example usage
distance_matrix = np.array([
    [0, 10, 15, 20, 30],
    [10, 0, 35, 25, 40],
    [15, 35, 0, 28, 22],
    [20, 25, 28, 0, 18],
    [30, 40, 22, 18, 0]
])

heuristics_matrix = heuristics_v2(distance_matrix)
print(heuristics_matrix)
