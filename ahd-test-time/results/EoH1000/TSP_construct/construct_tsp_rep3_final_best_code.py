import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Calculate the distances from the current node to all unvisited nodes and from the unvisited nodes to the destination node
    dists1 = distance_matrix[current_node, unvisited_nodes]
    dists2 = distance_matrix[unvisited_nodes, destination_node]

    # Apply sigmoid function to the distances
    sigmoid_dist1 = sigmoid(dists1)
    sigmoid_dist2 = sigmoid(dists2)

    # Normalize the sigmoid of the distances
    normalized_sigmoid_dist1 = sigmoid_dist1 / (sigmoid_dist2 * np.abs(dists2 - dists1) + np.finfo(np.float64).eps)
    normalized_sigmoid_dist2 = sigmoid_dist2 / (sigmoid_dist1 * np.abs(dists2 - dists1) + np.finfo(np.float64).eps)

    # Calculate the ratio of the normalized sigmoid of the distances
    ratio = normalized_sigmoid_dist1 / normalized_sigmoid_dist2

    # Select the node with the minimum ratio
    next_node = unvisited_nodes[np.argmin(ratio)]

    return next_node
