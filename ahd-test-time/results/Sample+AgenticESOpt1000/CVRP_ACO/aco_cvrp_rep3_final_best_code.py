# source: /home/zhi/Dynamic-Agent/cache/active_runs/aco_cvrp_train_sample_es_pop20_gen50_sigma0.001_alpha0.0005_aco_cvrp_sample_es_current_cosine_t1000_rep3_queue_a_gpu0_3_20260720_030717/results/pops_best/population_generation_50.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=50, samples=1000, sigma=0.001->0, alpha=0.0005
# task: aco_cvrp, rep: 3
# train_objective: 10.19867

import numpy as np
import math

def heuristics_v2(distance_matrix, coordinates, demands, capacity):
    n = len(distance_matrix)
    heuristics = np.ones((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                heuristics[i, j] = 0
            elif j == 0: # exclude routes to the depot node
                heuristics[i, j] = 0
            else:
                dist_demand_ratio = (1 - demands[j] / capacity)
                heuristics[i, j] = (1 / distance_matrix[i, j]) * dist_demand_ratio
                
    return heuristics

def distance_heuristic(node1, node2, coordinates):
    return np.linalg.norm(coordinates[node1] - coordinates[node2])

def stochastic_routing(distance_matrix, coordinates, demands, capacity):
    n = len(distance_matrix)
    candidates = np.arange(n)
    current_route = [0]  # start at the depot
    
    heuristics = heuristics_v2(distance_matrix, coordinates, demands, capacity)
    
    while len(current_route) < n: # explore all candidates
        current_node = current_route[-1]
        next_node = None
        max_heuristic = -np.inf
        for node in candidates:
            if node not in current_route and demands[node] < capacity:
                heuristic = heuristics[current_node, node]
                if heuristic > max_heuristic:
                    max_heuristic = heuristic
                    next_node = node
        if next_node!= None:
            current_route.append(next_node)
        else:
            break # no more feasible moves
    
    # Check if total demand in route exceeds the capacity
    visited_nodes = current_route[:-1]  # exclude the last node (returning to depot)
    visited_demands = [demands[node] for node in visited_nodes]
    total_demand = sum(visited_demands)
    if total_demand > capacity:
        excess_demand = total_demand - capacity
        # subtract most demanded node from route
        excess_removed_node = visited_demands.index(max(visited_demands))
        visited_nodes.remove(excess_removed_node)
        current_route.remove(excess_removed_node)
        
    current_distance = 0
    for i in range(len(current_route) - 1):
        current_distance += distance_heuristic(current_route[i], current_route[i + 1], coordinates)
        
    # If no feasible solutions are found, return a default answer
    if current_distance == 0:
        return -1
    else:
        return current_distance
