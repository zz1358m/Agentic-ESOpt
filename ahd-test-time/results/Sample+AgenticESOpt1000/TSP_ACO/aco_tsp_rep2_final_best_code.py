# source: /home/zhi/Agentic-ESOpt/cache/active_runs/aco_tsp_train_sample_es_pop20_gen50_sigma0.001_alpha0.0005_aco_tsp_sample_es_current_cosine_t1000_rep2_queue_a_gpu0_3_20260720_030717/results/pops_best/population_generation_50.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=50, samples=1000, sigma=0.001->0, alpha=0.0005
# task: aco_tsp, rep: 2
# train_objective: 5.92653

import numpy as np

def heuristics(distance_matrix):
    # A simple heuristic function, prioritizing edges with smaller distance
    return np.reciprocal(distance_matrix) ** 2

def heuristics_v2(distance_matrix, num_samples=1000):
    # Add a node for the starting point
    distance_matrix = np.concatenate((distance_matrix, [[0]], [0]), axis=0)
    distance_matrix = np.concatenate((distance_matrix, distance_matrix), axis=1)

    num_nodes = distance_matrix.shape[0]
    heuristics_val = heuristics(distance_matrix)

    # Find all possible edges
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            edges.append((i, j, heuristics_val[i, j]))

    # Stochastic solution sampling
    best_solutions = []
    for _ in range(num_samples):
        solution = np.arange(num_nodes) + 1
        np.random.shuffle(solution)

        # Ensure start and end node is node 0
        solution = np.insert(solution, 0, 0)
        solution = np.insert(solution, len(solution), 0)

        # Cost of the current solution
        cost = sum(distance_matrix[solution[i], solution[i + 1]] for i in range(len(solution) - 1))

        # Update heuristics value for the current solution
        heuristics_val_solution = sum(heuristics_val[solution[i], solution[i + 1]] for i in range(len(solution) - 1))

        # Save the best solutions
        best_solutions.append(heuristics_val_solution)

    # Average heuristics value of the best solutions
    heuristics_matrix = np.array([max(best_solutions)])
    return heuristics_matrix
