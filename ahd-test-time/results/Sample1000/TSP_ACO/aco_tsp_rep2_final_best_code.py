# source: /home/zhi/Dynamic-Agent/cache/active_runs/aco_tsp_train_sample_t1000_aco_tsp_sample_t1000_rep2_20260718_144918/results/pops_best/population_generation_50.json
# method: sample, prefix=1000, batch_size=20
# task: aco_tsp, rep: 2
# train_objective: 5.96875

import numpy as np
import random

def heuristics(distance_matrix):
    num_cities = len(distance_matrix)
    num_routes = 100  # number of initial routes to generate
    best_route = None
    best_route_distance = float('inf')
    heuristics_matrix = np.zeros(distance_matrix.shape, dtype=float)

    for _ in range(num_routes):
        route = [random.randint(0, num_cities - 1)]
        visited = set([route[-1]])
        distance = 0
        for _ in range(num_cities - 1):
            next_city = random.choice([i for i in range(num_cities) if i not in visited])
            distance += distance_matrix[route[-1]][next_city]
            route.append(next_city)
            visited.add(next_city)
        route.append(route[0])  # close the loop
        distance += distance_matrix[route[-1]][route[0]]
        if distance < best_route_distance:
            best_route_distance = distance
            best_route = route
        for i in range(1, len(route) - 1):
            u = route[i - 1]
            v = route[i]
            heuristics_matrix[u][v] = 1 / distance_matrix[u][v]
            heuristics_matrix[v][u] = 1 / distance_matrix[v][u]

    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):
            if distance_matrix[i][j] == 0:
                heuristics_matrix[i][j] = 0
            else:
                heuristics_matrix[i][j] /= 1 / (heuristics_matrix[i][j].sum() + 1e-9)

    return heuristics_matrix
