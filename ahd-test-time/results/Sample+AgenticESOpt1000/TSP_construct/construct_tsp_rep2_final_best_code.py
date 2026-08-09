# source: /home/zhi/Agentic-ESOpt/cache/active_runs/construct_tsp_train_sample_es_pop20_gen50_sigma0.001_alpha0.0005_construct_tsp_sample_es_reload_cosine_current_pop20_gen50_rep2_20260719_150222/results/pops_best/population_generation_50.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=50, samples=1000, sigma=0.001->0, alpha=0.0005
# task: construct_tsp, rep: 2
# train_objective: 6.66389

import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    {Our algorithm is a hybrid of nearest neighbor and furthest criterion methods. 
    It chooses the nearest node to the current node when there are unvisited neighbors, 
    but it also considers the nodes that are furthest from the destination node with a weighted priority.}
    """
    
    # Get the distances from the current node to all unvisited nodes
    distances = distance_matrix[current_node, unvisited_nodes]
    
    # Get the indices of unvisited nodes
    unvisited_indices = np.where(unvisited_nodes == 1)[0]
    
    # Get the distances from the destination node to all unvisited nodes
    destination_distances = distance_matrix[destination_node, unvisited_nodes]
    
    # Assign a priority to each unvisited node based on its distance to the destination node
    # and to the current node
    priority = (destination_distances - distances) / (distances + destination_distances + 1e-9)
    
    # Get the index of the node with the highest priority
    next_node_idx = np.argmax(priority)
    
    # Convert the index to the node ID
    next_node = unvisited_nodes[next_node_idx]
    
    return next_node

# Example usage
if __name__ == "__main__":
    # Example distance matrix
    distance_matrix = np.array([[0, 3, 4, 5, 6],
                               [3, 0, 2, 4, 5],
                               [4, 2, 0, 3, 2],
                               [5, 4, 3, 0, 3],
                               [6, 5, 2, 3, 0]])
    
    # Set the current node, destination node, and unvisited nodes
    current_node = 0
    destination_node = 4
    unvisited_nodes = np.array([1, 2, 3, 4])
    
    # Select the next node
    next_node = select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix)
    print(next_node)
