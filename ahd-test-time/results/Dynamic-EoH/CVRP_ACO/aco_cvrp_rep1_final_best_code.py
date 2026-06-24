# source: /home/zhi/Dynamic-Agent/cache/active_runs/aco_cvrp_train_es_sigma0.001_alpha0.0005_aco_cvrp_train_es_m1m2_sigma1e-3_alpha5e-4_rep1_m1m2_fixed_rerun_cvrp_only_20260610_044756/results/pops_best/population_generation_25.json

import numpy as np

def heuristics_v2(distance_matrix, coordinates, demands, capacity):
    n = len(coordinates)
    
    heuristics_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                heuristics_matrix[i, j] = 0.0
            elif i == 0 or j == 0:
                heuristics_matrix[i, j] = -np.inf
            else:
                # Calculate the node's contribution to the total demand of the vehicle
                total_demand = demands[j]
                contribution = 0
                if i > 0:
                    contribution = demands[i]
                remaining_capacity = max(0, capacity - contribution)
                
                # Calculate the forward attractiveness score of node j
                if distance_matrix[i, j]!= 0:
                    forward_attractiveness_score = 1 / (distance_matrix[i, j] / 10000)
                else:
                    forward_attractiveness_score = 0
                    
                # Calculate the backward attractiveness score of node j
                backward_attractiveness_score = 1 / ((np.linalg.norm(np.abs(np.array(coordinates[j]) - np.array(coordinates[i])))) / 10000)
                
                # Calculate the similarity score of node j to the starting location
                similarity_score = np.linalg.norm(np.array(coordinates[j]) - coordinates[0]) / (np.linalg.norm(coordinates[0]) / 10000)
                
                # Calculate the potential to visit node j
                potential_to_visit = (forward_attractiveness_score * similarity_score)
                
                # Calculate the potential to return from node j
                potential_to_return = (backward_attractiveness_score * similarity_score)
                
                # Calculate the node's influence to include it in the route
                influence_score = 1 - (total_demand / (remaining_capacity + total_demand)) * (remaining_capacity > 0)
                
                # Calculate the edge's promise based on the node's potential to visit, potential to return, and distance
                if distance_matrix[i, j]!= 0:
                    heuristics_score = potential_to_visit * potential_to_return * influence_score * (1 / (distance_matrix[i, j] / 10000))
                else:
                    heuristics_score = 0
                    
                heuristics_matrix[i, j] = heuristics_score
                
    return heuristics_matrix
