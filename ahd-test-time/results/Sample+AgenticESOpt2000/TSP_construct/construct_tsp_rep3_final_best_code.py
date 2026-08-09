# source: /home/zhi/Agentic-ESOpt/cache/active_runs/construct_tsp_train_sample_es_pop20_gen100_sigma0.001_alpha0.0005_construct_tsp_sample_es_current_cosine_t2000_rep3_queue_b_gpu4_7_20260720_030717/results/pops_best/population_generation_100.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=100, samples=2000, sigma=0.001->0, alpha=0.0005
# task: construct_tsp, rep: 3
# train_objective: 6.66389

import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # calculate distances from the destination node to all unvisited nodes
    remaining_distances = distance_matrix[destination_node, unvisited_nodes]
    
    # calculate weights for each unvisited node based on their remaining distances
    weights = 1 / (remaining_distances + np.finfo(float).eps)
    
    # calculate the sum of weights for the current node to all unvisited nodes
    weights_sum = np.sum(weights)
    
    # calculate the weighted sum of distances for the current node to all unvisited nodes
    weighted_distances = distance_matrix[current_node, unvisited_nodes] * weights / weights_sum
    
    # find the unvisited node with the minimum weighted distance
    next_node_index = np.argmin(weighted_distances)
    
    # get the ID of the unvisited node with the minimum weighted distance
    next_node = unvisited_nodes[next_node_index]
    
    return next_node
