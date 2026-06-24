# source: /home/zhi/Dynamic-Agent/cache/active_runs/aco_tsp_train_eoh_aco_tsp_train_eoh_rep3_eoh_aco_fastgen_esaligned_20260609_092237/results/pops_best/population_generation_25.json
# objective: 5.83479
# algorithm:  This new algorithm, called 'heuristics_v3', assigns probabilities to edges based on a combination of the relative total-degree of the end nodes, the cost of the edges themselves, the mean of degrees of all nodes, and a linear combination of their pairwise distance and degree similarity. 

import numpy as np

def heuristics_v3(distance_matrix):
    n = distance_matrix.shape[0]
    heuristics_matrix = np.zeros_like(distance_matrix)
    
    total_degrees = np.sum(distance_matrix, axis=1)
    degrees = np.sum(distance_matrix, axis=0)
    mean_degrees = np.mean(np.sum(distance_matrix, axis=1))
    neighbor_degrees = np.sum(distance_matrix, axis=1)
    
    for i in range(n):
        in_degree_simularity = np.sum((degrees > 0) & (degrees == degrees[i])) / np.sum(degrees > 0)
        out_degree_simularity = np.sum((neighbor_degrees > 0) & (neighbor_degrees == neighbor_degrees[i])) / np.sum(neighbor_degrees > 0)
        
        min_in_cost = np.min(distance_matrix[i])
        min_out_cost = np.min(distance_matrix[:, i])
        
        for j in range(n):
            if i!= j:
                edge_cost = distance_matrix[i, j]
                edge_popularity = np.sum(distance_matrix == edge_cost) / (n * n)
                if edge_cost == 0:
                    heuristics_matrix[i, j] = 0
                else:
                    heuristics_matrix[i, j] = (total_degrees[i] * total_degrees[j]) / (edge_cost ** 2) * (mean_degrees ** 0.5) * np.exp(-(edge_cost + (in_degree_simularity + out_degree_simularity) * edge_popularity) / (min_in_cost + min_out_cost))
    
    return heuristics_matrix
