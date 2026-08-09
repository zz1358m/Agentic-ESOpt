# source: /home/zhi/Agentic-ESOpt/cache/active_runs/aco_tsp_train_sample_es_pop20_gen50_sigma0.001_alpha0.0005_aco_tsp_sample_es_current_cosine_t1000_rep1_queue_b_gpu4_7_20260720_030717/results/pops_best/population_generation_50.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=50, samples=1000, sigma=0.001->0, alpha=0.0005
# task: aco_tsp, rep: 1
# train_objective: 5.95611

import numpy as np
import random

def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    heuristics_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i!= j:
                heuristics_matrix[i, j] = 1 / (distance_matrix[i, j] ** 2)
                heuristics_matrix[j, i] = heuristics_matrix[i, j]  # symmetry: a->b same as b->a
    return heuristics_matrix

def heuristics_v2(distance_matrix):
    n = distance_matrix.shape[0]
    num_samples = 1000  # number of randomly generated candidate tours
    best_tour = None
    
    # generate and evaluate random candidate tours
    for _ in range(num_samples):
        tour = list(range(n))
        random.shuffle(tour)
        tour.insert(0, tour[0])  # add initial city to the end of the tour
        
        cost = 0
        for i in range(1, n):
            cost += distance_matrix[tour[i - 1], tour[i]]
        
        if best_tour is None or cost < best_tour['cost']:
            best_tour = {'tour': tour, 'cost': cost}
    
    # build a matrix of probabilities based on heuristics and tour frequencies
    heuristics_matrix = heuristics(distance_matrix)
    probabilities = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i!= j:
                probabilities[i, j] = best_tour['cost'] - distance_matrix[i, j] * best_tour['tour'].count(i) - distance_matrix[j, i] * best_tour['tour'].count(j)
                probabilities[i, j] = probabilities[i, j] / (heuristics_matrix[i, j] * num_samples) + 1e-9  # add a small value to avoid division by zero
    
    return probabilities
