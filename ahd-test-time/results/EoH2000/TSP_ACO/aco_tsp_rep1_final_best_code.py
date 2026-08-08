# Archived from /home/zhi/Dynamic-Agent/cache/active_runs/aco_tsp_train_eoh_aco_tsp_train_eoh_rep1_orig_eoh_all6_k3_8gpu_20260713_142341/results/pops_best/population_generation_25.json
# run_id: aco_tsp_train_eoh_rep1_orig_eoh_all6_k3_8gpu_20260713_142341
# train_objective: 5.82279
# method: original EoH, population=10, generations=25, k=3 replicates

import numpy as np

def heuristics_v2(distance_matrix):
    n = len(distance_matrix)
    max_iter = 750
    learning_rate = 0.01
    min_learning_rate = 0.0001
    patience = 25
    current_patience = 0
    prev_score = 0
    best_heuristics_matrix = np.zeros((n, n))
    heuristics_matrix = np.zeros((n, n))
    threshold = n / 4  # new parameter
    num_edges = int(n * (n - 1) / 2)

    for i in range(max_iter):
        permutation = list(np.random.permutation(n))
        permutation = [permutation[0]] + permutation[1:]  # Make sure the cycle starts and ends at node 0
        scores = []
        for j in range(n - 1):
            j_counts = 0
            for k in range(n):
                if permutation[(j + 1) % n]!= k:
                    j_counts += 1
            avg_neighbor_m_density = j_counts / (n - 1)  # calculate average density
            diff_neighbor_m_density = np.abs(j_counts - threshold) / (n - 1)  # calculate absolute difference of neighbor density
            for m in range(n):
                if j!= m and permutation[(j + 1) % n]!= m:
                    edge_cost = 1 / distance_matrix[j, m]
                    if edge_cost!= 0:  
                        base_weight = edge_cost * (distance_matrix[j, m]**(-2) / np.sum(distance_matrix[:n-1, n-1]**(-2)))
                        heuristic = base_weight * diff_neighbor_m_density * (distance_matrix[j, m]**-1)
                    else:
                        heuristic = 0
                    heuristics_matrix[j, m] += learning_rate * heuristic
                    heuristics_matrix[m, j] += learning_rate * heuristic
            j_scores = np.log(n - 1) - np.log(1 + 1 / heuristics_matrix[j, :j])
            scores.append(np.mean(j_scores))

        # Calculate difference in heuristics matrix
        mean_scores = np.mean(scores)
        
        # Update the patience counter
        if mean_scores > prev_score and current_patience < patience:
            current_patience = 0
            previous_h_matrix = heuristics_matrix.copy()
        elif mean_scores <= prev_score:
            current_patience += 1
        
        # Update the learning rate using cosine annealing schedule
        alpha = 0.99
        learning_rate = min_learning_rate + 0.5 * (learning_rate - min_learning_rate) * (1 + np.cos((i + 1) * np.pi / max_iter))
        
        prev_score = mean_scores
        
        # Update edge inclusion probabilities
        for j in range(n):
            for m in range(n):
                if j!= m:
                    heuristics_matrix[j, m] *= 1 - (learning_rate - learning_rate * 0.99) # non-adaptive update
        # Store best heuristics if converged or early stopping
        if current_patience >= patience:
            break
        else:
            best_heuristics_matrix = heuristics_matrix.copy()

    return heuristics_matrix
