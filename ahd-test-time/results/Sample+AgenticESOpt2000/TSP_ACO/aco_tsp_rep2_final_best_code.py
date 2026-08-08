# source: /home/zhi/Dynamic-Agent/cache/active_runs/aco_tsp_train_sample_es_pop20_gen100_sigma0.001_alpha0.0005_aco_tsp_sample_es_current_cosine_t2000_rep2_queue_b_gpu4_7_20260720_030717/results/pops_best/population_generation_100.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=100, samples=2000, sigma=0.001->0, alpha=0.0005
# task: aco_tsp, rep: 2
# train_objective: 5.97428

import numpy as np
import random

def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    heuristics_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            heuristics_matrix[i, j] = 1 / distance_matrix[i, j] ** 2
    return heuristics_matrix

def heuristics_v2(distance_matrix):
    n = distance_matrix.shape[0]
    current_solution = np.ones((n,)) * -1
    population = [current_solution.copy()]
    iteration_limit = 1000
    iteration_number = 0
    
    while len(population) < 50 and iteration_number < iteration_limit:
        new_population = []
        for _ in range(100):
            current_solution = population[np.random.randint(len(population))].copy()
            if np.all(current_solution == -1):  # Ensure that each individual has visited all nodes
                continue
            index = np.random.randint(n)
            current_solution[index] = -1
            for i in range(n):
                if current_solution[i] == -1:
                    current_solution[i] = index
                    break
            while True:
                new_solution = current_solution.copy()
                idx1 = np.random.choice(n, replace=False)
                idx2 = np.random.choice(n, replace=False)
                current_solution[idx1] = idx2
                current_solution[idx2] = idx1
                if np.all(current_solution == -1) or (np.isin(idx1, np.where(current_solution == -1)).sum() < 2):
                    break
            new_population.append(current_solution)
        new_population = list(set(tuple(x) for x in new_population))
        population = new_population
        iteration_number += 1
    
    heuristics_matrix = np.zeros((n, n))
    for solution in population:
        for i in range(n):
            for j in range(i+1, n):
                if solution[i]!= -1 and solution[j]!= -1 and solution[i]!= solution[j]:
                    heuristics_matrix[solution[i], solution[j]] += 1
    
    return heuristics_matrix / np.max(heuristics_matrix)
