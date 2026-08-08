# source: /home/zhi/Dynamic-Agent/cache/active_runs/aco_cvrp_train_sample_t1000_aco_cvrp_sample_t1000_rep1_20260718_091608/results/pops_best/population_generation_50.json
# method: sample, prefix=1000, batch_size=20
# task: aco_cvrp, rep: 1
# train_objective: 9.68248

import numpy as np
from scipy.spatial import distance
from scipy.stats import norm
import math

def heuristics_v2(distance_matrix, coordinates, demands, capacity):
    n = len(coordinates)
    heuristics_matrix = np.zeros((n, n))

    for i in range(1, n):
        for j in range(1, n):
            if distance_matrix[i, j]!= 0 and distance_matrix[j, i]!= 0:
                # Step 1: Calculate the weight (1 if demand is within capacity, otherwise 0)
                if i!= j and demands[i] + demands[j] <= capacity:
                    weight = 1 / distance_matrix[i, j]
                else:
                    weight = 0

                # Step 2: Scale the weight based on the expected distance (uniform distribution)
                if distance_matrix[i, j]!= 0:
                    scale = math.sqrt(math.sqrt(len(demands)) / (distance_matrix[i, j] * distance_matrix[i, j]))
                else:
                    scale = 0

                # Step 3: Apply a heuristic to calculate the prior indicator
                heuristics_matrix[i, j] = weight * scale

    return heuristics_matrix
