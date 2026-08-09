# source: /home/zhi/Agentic-ESOpt/cache/active_runs/construct_tsp_train_sample_es_pop20_gen100_sigma0.001_alpha0.0005_construct_tsp_sample_es_current_cosine_t2000_rep2_queue_a_gpu0_3_20260720_030717/results/pops_best/population_generation_100.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=100, samples=2000, sigma=0.001->0, alpha=0.0005
# task: construct_tsp, rep: 2
# train_objective: 6.66389

import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node for the diversified nearest neighbor algorithm.

    Args:
    current_node: The ID of the current node.
    destination_node: The ID of the destination node.
    unvisited_nodes: A numpy array of IDs of unvisited nodes.
    distance_matrix: A numpy array of distances between nodes.

    Returns:
    next_node: The ID of the selected next node.
    """

    # Calculate the diversification scores
    diversification_scores = np.where(distance_matrix[current_node] == 0, 0, 
                                     distance_matrix[destination_node] / distance_matrix[current_node])

    # Find the next node with the highest diversification score in the unvisited nodes
    next_node_id = np.argmax(diversification_scores[unvisited_nodes])

    return unvisited_nodes[next_node_id]
