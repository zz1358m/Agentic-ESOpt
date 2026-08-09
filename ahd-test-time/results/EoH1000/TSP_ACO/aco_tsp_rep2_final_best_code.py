# source: /home/zhi/Agentic-ESOpt/cache/active_runs/aco_tsp_train_eoh_aco_tsp_train_eoh_rep2_eoh_aco_fastgen_esaligned_20260609_092237/results/pops_best/population_generation_25.json
# objective: 6.1567
# algorithm: The new algorithm, two-opt, uses stochastic sampling to generate a population of candidate solutions, and then applies a two-opt swap heuristic to iteratively improve promising solutions, tracking the frequency of each edge across the population and returning a prior probability for each edge based on this frequency.

import numpy as np
import random

def heuristics_v2(distance_matrix):
    num_nodes = len(distance_matrix)
    num_paths = 100
    max_length = 100
    population = []
    
    def random_path(nodes):
        path = list(nodes)
        random.shuffle(path)
        path = [0] + path
        path.append(0)
        return path
    
    def path_distance(path):
        distance = 0
        for i in range(len(path) - 1):
            distance += distance_matrix[path[i]][path[i + 1]]
        return distance
    
    for _ in range(num_paths):
        path = random_path(range(1, num_nodes))
        population.append(path)
    
    generations = 0
    while True:
        generations += 1
        new_population = population[:]
        num_improved = 0
        candidate_new_paths = []
        
        for i in range(num_paths):
            path = population[np.random.randint(len(population))]
            new_path = path[:]
            
            i, j = np.random.randint(1, len(new_path) - 1, 2)
            new_path[i:j+1] = new_path[i:j+1][::-1]
            
            distance = path_distance(new_path)
            candidate_new_paths.append((distance, new_path))
        
        # Sort candidate new paths
        candidate_new_paths.sort(key=lambda x: x[0])
        # Keep the best half
        candidate_new_paths = candidate_new_paths[:num_paths // 2]
        
        for new_distance, new_path in candidate_new_paths:
            index = np.random.randint(len(population))
            original_distance = path_distance(population[index])
            if new_distance < original_distance:
                new_population[index] = new_path
                num_improved += 1
        
        population = new_population
        if num_improved / num_paths < 0.01 and generations > max_length:
            break
    
    heuristics_matrix = np.zeros((num_nodes, num_nodes))
    for path in population:
        for i in range(len(path) - 1):
            start, end = path[i], path[i + 1]
            heuristics_matrix[start, end] += 1
    
    return heuristics_matrix / len(population)
