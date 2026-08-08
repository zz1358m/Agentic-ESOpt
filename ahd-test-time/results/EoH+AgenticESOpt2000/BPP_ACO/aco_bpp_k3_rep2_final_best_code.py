# Archived from /home/zhi/Dynamic-Agent/cache/active_runs/aco_bpp_train_es_sigma0.001_alpha0.0005_aco_bpp_train_es_full_reload_sigma1e-3_alpha5e-4_rep2_k3_decay_full_all6_3rep_reload8_20260705_121329/results/pops_best/population_generation_25.json
# run_id: aco_bpp_train_es_full_reload_sigma1e-3_alpha5e-4_rep2_k3_decay_full_all6_3rep_reload8_20260705_121329
# train_objective: 202.2
# m1m2_multiplier: 3.0
# sigma_schedule: cosine
# final_model_es_sigma: 0.000691341716182545

import numpy as np

def heuristics_v2(demand, capacity):
    n = len(demand)
    heuristics = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i!= j:
                if demand[i] + demand[j] <= capacity:
                    heuristics[i, j] = demand[i] * demand[j]
                    heuristics[j, i] = demand[i] * demand[j]
                else:
                    heuristics[i, j] = -1
                    heuristics[j, i] = -1
                    
    # Calculate the score for pairs of items that do not violate the capacity constraint
    for i in range(n):
        for j in range(n):
            if i!= j and heuristics[i, j]!= -1:
                max_demand_item = np.max([demand[i], demand[j]])
                score = - (max_demand_item - capacity) * max_demand_item + (demand[i] * demand[j]) 
                heuristics[i, j] = score
                heuristics[j, i] = score
                
    return heuristics

# Test the function with an example
demand = np.array([12, 11, 8, 6, 4])
capacity = 20
heuristics_matrix = heuristics_v2(demand, capacity)
print(heuristics_matrix)
