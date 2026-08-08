import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Remove current node and destination node from unvisited nodes
    unvisited_nodes = np.setdiff1d(unvisited_nodes, np.array([current_node, destination_node]))

    # Initialize minimum sum of deviations to a large value
    min_sum_deviations = float('inf')

    # Initialize next node to None
    next_node = None

    # Calculate the number of dimensions (number of unvisited nodes)
    num_dims = len(unvisited_nodes)

    # Iterate over unvisited nodes
    for node in unvisited_nodes:
        # Calculate deviations in each dimension
        deviations = distance_matrix[:, current_node] + distance_matrix[current_node, node] - distance_matrix[:, node] - distance_matrix[current_node, destination_node]

        # Calculate the sum of deviations
        sum_deviations = np.sum(deviations)

        # If the node has a smaller sum of deviations and is not the current node, update next node and minimum sum of deviations
        if sum_deviations < min_sum_deviations and node!= current_node:
            min_sum_deviations = sum_deviations
            next_node = node

    return next_node
