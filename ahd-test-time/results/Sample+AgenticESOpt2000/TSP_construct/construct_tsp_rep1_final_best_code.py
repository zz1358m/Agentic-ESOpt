# source: /home/zhi/Agentic-ESOpt/cache/active_runs/construct_tsp_train_sample_es_pop20_gen100_sigma0.001_alpha0.0005_construct_tsp_sample_es_current_cosine_t2000_rep1_queue_b_gpu4_7_20260720_030717/results/pops_best/population_generation_100.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=100, samples=2000, sigma=0.001->0, alpha=0.0005
# task: construct_tsp, rep: 1
# train_objective: 6.39225

import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Initialize variables
    visited_nodes = np.setdiff1d(unvisited_nodes, current_node)
    centroid_distance = np.zeros(len(visited_nodes))
    max_similarity = 0
    next_node = None

    # Calculate the centroid distance of each unvisited node
    for i, node in enumerate(visited_nodes):
        centroid_distance[i] = (distance_matrix[node, current_node] + 
                                np.sum([distance_matrix[node, n] for n in visited_nodes if n!= node]) + 
                                distance_matrix[destination_node, node]) / len(visited_nodes)

    # Select the node with the maximum centroid similarity
    max_similarity_idx = np.argmax(centroid_distance - distance_matrix[current_node, visited_nodes])
    next_node = visited_nodes[max_similarity_idx]

    return next_node
