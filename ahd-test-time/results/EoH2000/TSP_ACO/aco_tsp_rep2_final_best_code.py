# Archived from /home/zhi/Dynamic-Agent/cache/active_runs/aco_tsp_train_eoh_aco_tsp_train_eoh_rep2_orig_eoh_all6_k3_8gpu_20260713_142341/results/pops_best/population_generation_25.json
# run_id: aco_tsp_train_eoh_rep2_orig_eoh_all6_k3_8gpu_20260713_142341
# train_objective: 5.85615
# method: original EoH, population=10, generations=25, k=3 replicates

import numpy as np

def heuristics_v3(distance_matrix, num_iter=10000, temperature=100, degree_weight=0.5):
    """
    This function generates a heat map indicating the probability of each edge being part of an optimal solution.

    Parameters:
    - distance_matrix (2D numpy array): A matrix representing the distances between city locations.
    - num_iter (int): The number of iterations to run the simulation (default=10000).
    - temperature (float): The initial temperature of the simulated annealing process (default=100).
    - degree_weight (float): The weight given to node degrees in the score function (default=0.5).

    Returns:
    - heuristics_matrix (2D numpy array): A matrix representing the probability of each edge being part of an optimal solution.
    """

    # Initialize the heat map matrix with equal probabilities for all edges
    heat_map = np.ones((distance_matrix.shape[0], distance_matrix.shape[0]))

    # Calculate the degree matrix (number of connections to each node)
    degree_matrix = np.sum(distance_matrix, axis=0)

    # Initialize the current best distance and solution
    best_distance = float('inf')
    best_solution = None

    for _ in range(num_iter):
        # Generate a random permutation of all city locations
        permutation = np.random.permutation(distance_matrix.shape[0])

        # Calculate the total distance for the current permutation
        current_distance = 0
        for i in range(permutation.shape[0]):
            current_distance += distance_matrix[permutation[i], permutation[(i+1)%permutation.shape[0]]]

        # Check if the current solution is better than the best solution found so far
        if current_distance < best_distance:
            best_distance = current_distance
            best_solution = permutation

        # Update the heat map based on the current solution
        for i in range(permutation.shape[0]):
            for j in range(permutation.shape[0]):
                # Score function: favors solutions with more connections to nodes with high edge weights
                edge_weight = distance_matrix[i, j]
                score = edge_weight + degree_matrix[i] + degree_matrix[j]
                score = np.exp(-temperature * edge_weight / (best_distance + score))
                heat_map[i, j] *= score ** degree_weight

        # Perform simulated annealing to adjust the temperature
        temperature *= 0.99

    # Calculate the prior probability of each edge being part of an optimal solution
    total_probabilities = heat_map.sum(axis=1, keepdims=True)
    heuristics_matrix = heat_map / total_probabilities

    return heuristics_matrix
