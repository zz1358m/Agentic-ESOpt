# source: /home/zhi/Agentic-ESOpt/cache/active_runs/aco_tsp_train_eoh_aco_tsp_train_eoh_rep1_eoh_aco_fastgen_esaligned_20260609_092237/results/pops_best/population_generation_25.json
# objective: 5.80965
# algorithm: This new algorithm introduces a decay function to the novelty penalty, allowing it to decay exponentially with distance, with a high decay rate for short distances, and then increases it for long distances, giving more importance to the shortest paths.

import numpy as np

def heuristics_v2(distance_matrix):
    num_nodes = distance_matrix.shape[0]
    heuristics_matrix = np.zeros(distance_matrix.shape)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i!= j:
                if distance_matrix[i, j] == 0:
                    weight = np.inf
                else:
                    distance_ratio = distance_matrix[i, j] ** 3
                    novelty_penalty = np.exp(-distance_matrix[i, j])
                    weight = 1 / distance_matrix[i, j] * novelty_penalty / distance_ratio
                heuristics_matrix[i, j] = weight
                heuristics_matrix[j, i] = weight

    return heuristics_matrix
