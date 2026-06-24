import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    weights = np.zeros(len(unvisited_nodes))
    
    for i, node in enumerate(unvisited_nodes):
        if node!= destination_node:
            weights[i] = 1 / distance_matrix[current_node, node]
            weights[i] *= np.sqrt(distance_matrix[node, destination_node] / distance_matrix[current_node, destination_node])
            weights[i] *= np.mean([distance_matrix[node, unvisited_node] for unvisited_node in unvisited_nodes])
    
    weights /= np.sum(weights)
    
    next_node_index = np.argmax(weights)
    
    next_node = unvisited_nodes[next_node_index]
    
    return next_node
