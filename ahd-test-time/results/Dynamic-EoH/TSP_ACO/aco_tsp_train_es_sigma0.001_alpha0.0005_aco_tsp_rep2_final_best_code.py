import numpy as np

def heuristics_v2(distance_matrix):
    num_cities = distance_matrix.shape[0]
    
    # Apply a logarithmic function to the weights of the edges to create a decay factor
    decay_factors = np.log(1 + distance_matrix)
    
    # Multiply the original weights by the decay factor
    power_matrix = distance_matrix * (1 / (1 + distance_matrix))
    
    # Calculate the scaling factor for each edge
    scaling_factors = np.reciprocal((power_matrix) ** 2)
    
    # Normalize the edge weights by the scaling factors along each row
    heuristics_matrix = np.where(power_matrix == 0, scaling_factors, scaling_factors * (power_matrix ** -2 / np.sum(power_matrix ** -2)))
    
    # Normalize to sum to 1 along each row
    heuristics_matrix /= np.sum(heuristics_matrix, axis=1, keepdims=True)
    
    return heuristics_matrix
