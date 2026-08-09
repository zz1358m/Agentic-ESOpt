# source: /home/zhi/Agentic-ESOpt/cache/active_runs/aco_bpp_train_sample_es_pop20_gen50_sigma0.001_alpha0.0005_aco_bpp_sample_es_current_cosine_t1000_rep2_queue_a_gpu0_3_20260720_030717/results/pops_best/population_generation_50.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=50, samples=1000, sigma=0.001->0, alpha=0.0005
# task: aco_bpp, rep: 2
# train_objective: 202.4

import numpy as np

def heuristics_v2(demand, capacity):
    n = len(demand)
    bins = []
    heuristics = np.zeros((n, n))
    
    items = sorted(range(n), key=lambda i: demand[i], reverse=True)
    
    for i in range(n):
        item = items[i]
        placed = False
        
        for j, bin in enumerate(bins):
            total_demand = sum([demand[i] for i in bin])
            if total_demand + demand[item] <= capacity:
                heuristics[item, bin] = heuristics[bin, item] = 1
                bins[j].append(item)
                placed = True
                break
        
        if not placed:
            bins.append([item])
    
    return np.array(heuristics)
