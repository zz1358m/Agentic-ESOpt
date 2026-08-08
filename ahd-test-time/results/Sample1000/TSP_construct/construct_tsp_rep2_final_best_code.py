# source: /home/zhi/Dynamic-Agent/cache/active_runs/construct_tsp_train_sample_t1000_construct_tsp_sample_t1000_rep2_20260718_144918/results/pops_best/population_generation_50.json
# method: sample, prefix=1000, batch_size=20
# task: construct_tsp, rep: 2
# train_objective: 7.00437

import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    {
        # New algorithm: 'Next Node Selection using Max Similarity of Adjacent Nodes' (NSAN)
        # This algorithm chooses the next node with the maximum similarity of its adjacent nodes' unvisited nodes.
    }

    # Calculate similarity of adjacent nodes
    adjacent_nodes_similarity = np.zeros_like(unvisited_nodes)
    for i in range(len(unvisited_nodes)):
        adjacent_nodes = set([index for index in range(len(distance_matrix)) if index!= current_node and index!= destination_node])
        intersection = adjacent_nodes.intersection(unvisited_nodes)
        similarity = len(intersection) / len(adjacent_nodes)
        adjacent_nodes_similarity[i] = similarity

    # Find the node with the maximum similarity
    next_node_index = np.argmax(adjacent_nodes_similarity[np.isfinite(adjacent_nodes_similarity)])
    next_node = unvisited_nodes[next_node_index]

    # Remove the selected node from unvisited_nodes
    unvisited_nodes = np.delete(unvisited_nodes, next_node_index)

    return next_node
