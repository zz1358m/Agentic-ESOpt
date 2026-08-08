# source: /home/zhi/Dynamic-Agent/cache/active_runs/aco_cvrp_train_sample_es_pop20_gen100_sigma0.001_alpha0.0005_aco_cvrp_sample_es_current_cosine_t2000_rep1_queue_a_gpu0_3_20260720_030717/results/pops_best/population_generation_100.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=100, samples=2000, sigma=0.001->0, alpha=0.0005
# task: aco_cvrp, rep: 1
# train_objective: 9.83064

import numpy as np

def heuristics_v2(distance_matrix, coordinates, demands, capacity):
    n = len(coordinates)
    node_positions = [[node for node in coordinates]]

    # Step 1: Initialize heuristics matrix with zeros
    heuristics_matrix = np.zeros((n, n))

    # Step 2: Calculate the number of vehicles required
    num_vehicles = np.ceil(np.sum(demands) / capacity).astype(int)

    # Step 3: Initialize a list to store the list of nodes per vehicle
    nodes_per_vehicle = [[] for _ in range(int(num_vehicles))]

    # Step 4: Sort the nodes by demand in descending order
    sorted_nodes = np.argsort(-demands)

    # Step 5: Allocate nodes to vehicles in a greedy manner
    for node in sorted_nodes:
        if len(nodes_per_vehicle[0]) == capacity // 2:
            # Move node to the next vehicle if it has reached capacity
            next_vehicle = np.argmin([len(n_nodes) for n_nodes in nodes_per_vehicle]) if nodes_per_vehicle else 0
            nodes_per_vehicle[next_vehicle].append(node)
        else:
            nodes_per_vehicle[0].append(node)

    # Step 6: For each vehicle, calculate a cost reduction for each route
    for v in range(num_vehicles):
        demand = 0
        nodes = nodes_per_vehicle[v]
        last_node = -1
        for node in nodes:
            demand += demands[node]
            # We use a greedy approach to assign a high heuristic to edges that connect two nodes with high demands
            if demand >= capacity:
                break
            if last_node!= -1:
                heuristics_matrix[last_node, node] = (demands[node] + demands[last_node]) / (distance_matrix[last_node, node] ** 2)
            last_node = node
            heuristics_matrix[node, last_node] = (demands[node] + demands[last_node]) / (distance_matrix[last_node, node] ** 2)

    # Step 7: For edges that are not included in any vehicle's route, assign a 0 heuristic
    for i in range(n):
        for j in range(n):
            if heuristics_matrix[i, j] == 0:
                if i!= j and demands[i] > 0 and demands[j] > 0:
                    heuristics_matrix[i, j] = (demands[i] + demands[j]) / (distance_matrix[i, j] ** 2)

    return heuristics_matrix
