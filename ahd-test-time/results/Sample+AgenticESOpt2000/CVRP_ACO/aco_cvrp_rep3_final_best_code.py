# source: /home/zhi/Agentic-ESOpt/cache/active_runs/aco_cvrp_train_sample_es_pop20_gen100_sigma0.001_alpha0.0005_aco_cvrp_sample_es_current_cosine_t2000_rep3_queue_a_gpu0_3_20260720_030717/results/pops_best/population_generation_100.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=100, samples=2000, sigma=0.001->0, alpha=0.0005
# task: aco_cvrp, rep: 3
# train_objective: 9.39352

# evaluation_repairs: exp_log_upper_clip_60

import numpy as np
import math
from collections import deque
import random

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = len(coordinates)
    heuristics_matrix = np.full((n, n), 0.0)
    
    def get_distance(i, j):
        return distance_matrix[i, j]
    
    def heuristic(i, j):
        # Calculate the total demand of the potential path from i to j
        total_demand = demands[i] + demands[j]
        path_demand = 0
        current_node = j
        visited = set([j])
        while True:
            for next_node in range(n):
                if next_node!= current_node and next_node not in visited and demands[next_node] + path_demand <= capacity:
                    path_demand += demands[next_node]
                    current_node = next_node
                    visited.add(current_node)
                    if next_node == i:
                        break
            if len(visited) == n:
                break
            path_demand += demands[current_node]
            current_node = current_node
            visited.add(current_node)
        return path_demand
    
    def calculate_heuristics(i, j):
        demand = demands[i]
        node = i
        score = 0
        visited = set([node])
        while True:
            for next_node in range(n):
                if next_node!= node and next_node not in visited and demand + demands[next_node] <= capacity:
                    score -= get_distance(node, next_node)
                    score -= get_distance(node, next_node)
                    demand += demands[next_node]
                    node = next_node
                    visited.add(next_node)
                    if len(visited) == n:
                        score += get_distance(node, 0)
                        score -= get_distance(node, 0)
                        return score
                    elif node == next_node:
                        return score
                    else:
                        node = next_node
            break
        return score
    
    for i in range(1, n):
        for j in range(1, n):
            heuristics_matrix[i, j] = np.exp(np.minimum(-calculate_heuristics(i, j) / get_distance(i, j), 60.0))
    
    return heuristics_matrix
