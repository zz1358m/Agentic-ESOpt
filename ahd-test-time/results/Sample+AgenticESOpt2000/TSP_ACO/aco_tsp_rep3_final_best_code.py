# source: /home/zhi/Dynamic-Agent/cache/active_runs/aco_tsp_train_sample_es_pop20_gen100_sigma0.001_alpha0.0005_aco_tsp_sample_es_current_cosine_t2000_rep3_queue_b_gpu4_7_20260720_030717/results/pops_best/population_generation_100.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=100, samples=2000, sigma=0.001->0, alpha=0.0005
# task: aco_tsp, rep: 3
# train_objective: 5.91751

import numpy as np

def heuristics(distance_matrix):
    # Distance to neighbor nodes from given node
    n = len(distance_matrix)
    heuristics_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i!= j:
                heuristics_matrix[i, j] = 1 / (distance_matrix[i, j] ** 2)
    return heuristics_matrix

def heuristics_v2(distance_matrix):
    population_size = 100
    num_generations = 500
    mutation_rate = 0.1
    n = len(distance_matrix)
    
    # Initial sampling of random tours
    population = []
    for _ in range(population_size):
        tour = list(range(n))
        np.random.shuffle(tour)
        population.append(tour)
    
    for _ in range(num_generations):
        # Select k most promising tours
        k = 5
        best_tours = sorted(population, key=lambda x: sum([distance_matrix[i-1][x[i]] for i in range(len(x))]) if len(x) > 1 else 0, reverse=True)[:k]
        
        # Combine tours to form initial population
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = np.random.choice(best_tours, 2, replace=False)
            child = parent1[:len(parent1)//2] + parent2[len(parent2)//2:]
            if len(child)!= len(set(child)):
                continue
            new_population.append(child)
        
        # Tournament selection, edge swaps, and insertion mutations
        population = new_population
        for _ in range(population_size):
            parent1, parent2 = np.random.choice(population, 2, replace=False)
            if np.random.rand() < mutation_rate:
                # Edge swap
                if len(parent1) > 2:
                    i, j = np.random.randint(0, len(parent1)-1, 2)
                    parent1[i], parent1[j] = parent1[j], parent1[i]
                # Insertion mutation
                else:
                    i = np.random.randint(0, len(parent1)-1)
                    new_node = np.random.choice([x for x in population[0] if x not in parent1])
                    parent1 = parent1[:i] + [new_node] + parent1[i:]
            population.remove(parent1)
            population.append(parent1)
    
    # Final evaluation and return
    heuristics_matrix = heuristics(distance_matrix)
    tour = sorted([x for x in population], key=lambda x: sum([heuristics_matrix[i-1][x[i]] for i in range(len(x)) if i > 0]) if len(x) > 1 else 0, reverse=True)[0]
    return heuristics_matrix[tour, :][:, tour]
