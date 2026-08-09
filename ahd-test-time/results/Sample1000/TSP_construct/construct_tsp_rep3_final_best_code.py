# source: /home/zhi/Agentic-ESOpt/cache/active_runs/construct_tsp_train_sample_t1000_construct_tsp_sample_t1000_rep3_20260718_144918/results/pops_best/population_generation_50.json
# method: sample, prefix=1000, batch_size=20
# task: construct_tsp, rep: 3
# train_objective: 6.60917

import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    {"""Select the next node based on the weighted sum of distances between current node and all unvisited nodes, 
    the similarity of the node's degree and the maximum degree of unvisited nodes, and the node's distance to the destination."""
    }

    # Calculate distances to all unvisited nodes
    distances = distance_matrix[current_node] - distance_matrix[current_node, current_node]
    distances[~np.isin(range(len(distance_matrix)), unvisited_nodes)] = np.inf
    
    # Calculate degrees of nodes
    degrees = np.sum(distance_matrix, axis=0)
    max_degree = np.max(degrees[unvisited_nodes])
    
    # Calculate weighted similarities of node degrees
    similarities = degrees[unvisited_nodes] / max_degree
    
    # Calculate the weighted sum of distances and similarities
    weights = distances[unvisited_nodes] + 0.5 * (1 / similarities)
    
    # Select the next node with the minimum weighted sum
    next_node = unvisited_nodes[np.argmin(weights)]
    
    # Update the unvisited nodes
    unvisited_nodes = np.setdiff1d(unvisited_nodes, next_node)
    
    return next_node
