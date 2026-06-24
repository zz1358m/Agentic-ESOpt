import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    distances_from_current = distance_matrix[current_node, unvisited_nodes]
    distances_to_destination = distance_matrix[unvisited_nodes, destination_node]
    
    diff_distances_current = np.abs(distances_from_current)
    diff_distances_destination = np.abs(distances_to_destination)
    
    thresholds = (1 + diff_distances_current) / (1 + diff_distances_destination)
    
    next_node_index = np.argmin(thresholds)
    
    if thresholds[next_node_index] > thresholds[np.argmax(thresholds)]:
        return unvisited_nodes[np.argmin(distances_from_current)]
    else:
        return unvisited_nodes[next_node_index]
