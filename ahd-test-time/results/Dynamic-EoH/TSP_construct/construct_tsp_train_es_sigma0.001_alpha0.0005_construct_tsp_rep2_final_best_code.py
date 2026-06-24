import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit based on the Variation of Neighborhood Dominance Algorithm.

    Parameters:
    current_node (int): The ID of the current node.
    destination_node (int): The ID of the destination node.
    unvisited_nodes (numpy array): An array of IDs of unvisited nodes.
    distance_matrix (numpy array): The distance matrix of nodes.

    Returns:
    next_node (int): The ID of the next node to visit.
    """

    # Calculate the distance to the destination node from the unvisited nodes
    destination_distances = distance_matrix[:, destination_node]

    # Initialize the minimum ratio and the next node
    min_ratio = float('inf')
    next_node = None

    # Iterate over the unvisited nodes
    for node in unvisited_nodes:
        # Calculate the distance to the current node and to the destination node from the candidate node
        distance_to_current = distance_matrix[node, current_node]
        distance_to_destination = destination_distances[node]

        # Calculate the ratio and check if it's the minimum ratio found so far
        ratio = distance_to_current / distance_to_destination
        if ratio < min_ratio:
            min_ratio = ratio
            next_node = node

        # In case of a tie, favor nodes with shorter distances to the destination node
        elif ratio == min_ratio and distance_to_destination < destination_distances[np.argmin(destination_distances)]:
            next_node = node

    # Remove the current and next node from the unvisited nodes list
    unvisited_nodes = np.delete(unvisited_nodes, np.where(unvisited_nodes == current_node))
    unvisited_nodes = np.delete(unvisited_nodes, np.where(unvisited_nodes == next_node))

    return next_node
