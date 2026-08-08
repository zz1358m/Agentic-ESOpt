# source: /home/zhi/Dynamic-Agent/cache/active_runs/aco_tsp_train_sample_t2000_aco_tsp_sample_t2000_from_rep3_20260718_145101/results/pops_best/population_generation_100.json
# method: sample, prefix=2000, batch_size=20
# task: aco_tsp, rep: 3
# train_objective: 5.88644

import numpy as np
import random

def heuristics(distance_matrix):
    {
        # Randomly assign a high probability of being selected for the most direct paths and low probability for non-direct paths
    }
    num_nodes = distance_matrix.shape[0]
    heuristics_matrix = np.ones((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i!= j:
                heuristics_matrix[i, j] = 1 / (distance_matrix[i, j] ** 2)
    return heuristics_matrix

def heuristics_v2(distance_matrix):
    heuristics_matrix = heuristics(distance_matrix)
    num_nodes = heuristics_matrix.shape[0]
    
    # Initialize current node
    current_node = 0
    
    # Initialize solution path
    path = [0] * num_nodes
    path[0] = 0
    unvisited_nodes = list(range(1, num_nodes))
    
    for i in range(1, num_nodes):
        # Choose a node with high heuristic value
        max_heuristic_value = 0
        max_heuristic_node = None
        for node in unvisited_nodes:
            heuristic_value = heuristics_matrix[current_node, node]
            if heuristic_value > max_heuristic_value:
                max_heuristic_value = heuristic_value
                max_heuristic_node = node
        path[i] = max_heuristic_node
        unvisited_nodes.remove(max_heuristic_node)
        current_node = max_heuristic_node
    
    # Add back to the start node
    path = np.insert(path, 0, 0)
    
    # Create the heuristic matrix by selecting the selected edges from the original distance matrix
    heuristics_matrix_selected = distance_matrix
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if i == 0 and j == num_nodes - 1 or j == path[i] + 1 and i == path[j] - 1:
                heuristics_matrix_selected[i, j] = 0
                heuristics_matrix_selected[j, i] = 0
    
    return heuristics_matrix_selected
