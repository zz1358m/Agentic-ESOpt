# source: /home/zhi/Agentic-ESOpt/cache/active_runs/aco_tsp_train_sample_es_pop20_gen50_sigma0.001_alpha0.0005_aco_tsp_sample_es_current_cosine_t1000_rep3_queue_b_gpu4_7_20260720_030717/results/pops_best/population_generation_50.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=50, samples=1000, sigma=0.001->0, alpha=0.0005
# task: aco_tsp, rep: 3
# train_objective: 5.92362

import numpy as np

def heuristics(distance_matrix):
    # Calculate a heuristic for the importance of each edge (inverse distance and logarithm to avoid negative infinity)
    heuristics_matrix = np.where(distance_matrix!= 0, 1 / (distance_matrix * np.log(1 + distance_matrix)), np.inf)
    return heuristics_matrix

def heuristics_v2(distance_matrix):
    """
    Implementing the stochastic solution sampling heuristic for the Traveling Salesman Problem
    {Uses sampling to traverse the distance matrix, assign a random selection probability to each edge, iteratively sample the matrix with replacement according to the probabilities and calculate the expected shortest path, repeat the sampling steps, finally return a matrix indicating how promising each edge is to include in a solution}
    """
    
    num_repetitions = 10  # Number of repetitions of the sampling process
    matrix_size = distance_matrix.shape[0]
    
    # Initialize the selection matrix with uniform probabilities
    selection_probabilities = np.ones((matrix_size, matrix_size))
    np.fill_diagonal(selection_probabilities, 0)  # Avoid self-loops
    
    # Calculate the probability sums for normalization
    selection_sums = np.sum(selection_probabilities, axis=1)
    
    # Normalize the selection probabilities
    selection_probabilities /= selection_probabilities
    
    heuristic_sum = np.zeros((matrix_size, matrix_size))
    for _ in range(num_repetitions):
        random_sample = np.random.choice(matrix_size, matrix_size, p=np.ravel(selection_probabilities))
        sample_matrix = distance_matrix[random_sample, :]
        sample_matrix[:, random_sample] = sample_matrix
        heuristic_sum += sample_matrix / num_repetitions
    
    heuristics_matrix = 1 / heuristic_sum
    heuristics_matrix = np.nan_to_num(heuristics_matrix)
    heuristics_matrix[heuristics_matrix == 0] = np.inf  # Convert zero values to infinite
    return heuristics_matrix
