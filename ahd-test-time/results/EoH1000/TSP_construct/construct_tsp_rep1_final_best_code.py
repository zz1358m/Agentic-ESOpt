import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Initialize the next node as None
    next_node = None
    
    # Initialize the maximum difference and the index of the node with such a maximum
    max_diff = -np.inf
    next_node_idx = -1
    
    # Calculate the distance from the destination node to all unvisited nodes
    dest_dists = distance_matrix[unvisited_nodes, destination_node]
    
    # Calculate the distance from the current node to all unvisited nodes
    curr_dists = distance_matrix[current_node, unvisited_nodes]
    
    # Calculate the average distance from the destination node to all unvisited nodes
    avg_dest_dist = np.mean(dest_dists)
    
    # Calculate the priority for each unvisited node
    priorities = (dest_dists - curr_dists) / (dest_dists + avg_dest_dist)
    
    # Update the maximum difference and the index of the node with such a maximum
    for i, node in enumerate(unvisited_nodes):
        # If the node is the current node or the destination node, skip it
        if node == current_node or node == destination_node:
            continue
        # Update the maximum difference and the index of the node with such a maximum
        if priorities[i] > max_diff:
            max_diff = priorities[i]
            next_node_idx = node
    
    # Update the next node as the node with the maximum priority
    next_node = next_node_idx
    
    return next_node
