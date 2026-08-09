# source: /home/zhi/Agentic-ESOpt/cache/active_runs/aco_tsp_train_es_sigma0.001_alpha0.0005_aco_tsp_train_es_full_reload_sigma1e-3_alpha5e-4_rep2_rerun_agentic_esopt_k3_kp_tspaco_3rep_8gpu_20260716_120821/results/pops_best/population_generation_25.json
# run_id: aco_tsp_train_es_full_reload_sigma1e-3_alpha5e-4_rep2_rerun_agentic_esopt_k3_kp_tspaco_3rep_8gpu_20260716_120821
# train_objective: 5.82719
# m1m2_multiplier: 3.0

import numpy as np
import random

def heuristics_v3(distance_matrix):
    n = distance_matrix.shape[0]
    
    # Initialize the heuristics matrix with a value of 0
    heuristics_matrix = np.zeros((n, n))
    
    # Define node degrees in the graph
    node_degrees = np.sum(distance_matrix, axis=0)
    
    for i in range(n):
        for j in range(n):
            if i!= j:
                heuristics_matrix[i, j] = (10 / (distance_matrix[i, j] ** 3)) / (distance_matrix[i, j] + 1) * (node_degrees[i] * node_degrees[j] / (np.max(node_degrees) ** 2))
                
    return heuristics_matrix

def simulated_annealing(distance_matrix, max_it=100, T_init=1000, T_cool=0.999, accept_rate=0.1):
    n = distance_matrix.shape[0]
    path = np.arange(n)
    
    np.random.shuffle(path)
    path = np.append(path, path[0])
    
    heuristics_matrix = heuristics_v3(distance_matrix)
    current_path = path.copy()
    best_path = path.copy()
    
    T = T_init
    
    for _ in range(max_it):
        for i in range(int(n * accept_rate)):
            neighbor = current_path.copy()
            k = random.randint(1, path.shape[0]-2)
            i1, i2 = neighbor[k-1], neighbor[k]
            neighbor = np.delete(neighbor, np.array([k-1, k]))
            neighbor = np.append(neighbor, i2)
            neighbor = np.append(neighbor, i1)
            neighbor = np.append(neighbor, neighbor[0])
            
            # Calculate path length
            cost = np.sum(distance_matrix[neighbor, :][:, neighbor])
            
            # Calculate change in heuristic
            change = (10 * 1/(distance_matrix[neighbor, neighbor[:, None]] + distance_matrix[neighbor[:, None], neighbor])).sum() - (10 * 1/(distance_matrix[current_path, current_path[:, None]] + distance_matrix[current_path[:, None], current_path])).sum()
            
            # Acceptance criterion
            if change < 0 or random.random() < np.exp(-change/T):
                current_path = neighbor
                
        # Update temperature and check for improvement
        T *= T_cool
        if T < 0.01:
            break
        
        # Path relinking
        if random.random() < 0.2:
            k = random.randint(1, current_path.shape[0]-1)
            start = current_path[:k]
            end = current_path[k:]
            best_neighbor = current_path.copy()
            while True:
                k = random.randint(1, best_path.shape[0]-1)
                start = best_path[:k]
                end = best_path[k:]
                new_path = np.concatenate((start, end), axis=0)
                new_path = np.append(new_path, new_path[0])
                new_path = np.append(new_path, new_path[-2])
                
                # Calculate path length
                cost = np.sum(distance_matrix[new_path, :][:, new_path])
                
                # Calculate change in heuristic
                change = (10 * 1/(distance_matrix[new_path, new_path[:, None]] + distance_matrix[new_path[:, None], new_path])).sum()
