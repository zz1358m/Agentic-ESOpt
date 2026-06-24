import numpy as np

def heuristics_v3(distance_matrix):
    n = len(distance_matrix)
    mean_to_origin = np.mean(np.sum(distance_matrix, axis=1) - distance_matrix[0])
    median_to_origin = np.median(np.sum(distance_matrix, axis=1) - distance_matrix[0])
    total_distance_to_origin = np.sum(np.sum(distance_matrix, axis=1) - distance_matrix[0])
    
    heuristics_matrix = np.zeros((n, n))
    for i in range(n):
        row_sum = np.sum(distance_matrix[i, :]) - distance_matrix[0, i]
        for j in range(n):
            if i!= j and distance_matrix[i, j]!= 0:
                weight_to_i = (total_distance_to_origin + np.sum(distance_matrix[i, :])) / n
                weight_to_j = (total_distance_to_origin + np.sum(distance_matrix[j, :])) / n
                heuristics_matrix[i, j] = distance_matrix[i, j]**(-3) / (row_sum * weight_to_i * weight_to_j / (median_to_origin ** 1.5))
            else:
                heuristics_matrix[i, j] = 1e10
    return heuristics_matrix
