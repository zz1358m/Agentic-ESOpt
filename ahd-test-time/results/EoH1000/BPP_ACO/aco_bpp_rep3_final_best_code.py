# source: /home/zhi/Agentic-ESOpt/cache/active_runs/aco_bpp_train_eoh_aco_bpp_train_eoh_rep3_rerun_eoh_k1_bpp_3rep_8gpu_20260716_120821/results/pops_best/population_generation_25.json
# run_id: aco_bpp_train_eoh_rep3_rerun_eoh_k1_bpp_3rep_8gpu_20260716_120821
# train_objective: 202.2
# m1m2_multiplier: 1

import numpy as np

def heuristics_v2(demand, capacity):
    n = len(demand)
    sorted_indices = np.argsort(demand)[::-1]
    demand_sorted = demand[sorted_indices]
    
    clusters = {}
    for i in range(n):
        added = False
        for key, cluster in clusters.items():
            if np.sum(demand_sorted[cluster]) + demand_sorted[i] <= capacity:
                clusters[key].append(i)
                added = True
                break
        if not added:
            clusters[i] = [i]
    
    heuristics_matrix = np.zeros((n, n))
    
    for i in clusters:
        for j in clusters[i]:
            for pair in clusters:
                if pair!= i and j in clusters[pair]:
                    heuristics_matrix[sorted_indices[j], sorted_indices[j]] = 1
        
        for j in clusters[i]:
            for k in clusters[i]:
                if j!= k:
                    heuristics_matrix[sorted_indices[j], sorted_indices[k]] = (demand_sorted[i] - abs(demand_sorted[j] - demand_sorted[k])) / demand_sorted[i]
    
    for i in range(n):
        for j in range(n):
            if demand_sorted[i] > 0 and j!= i:
                heuristics_matrix[sorted_indices[j], sorted_indices[i]] = np.minimum(heuristics_matrix[sorted_indices[j], sorted_indices[i]], 1)
    
    heuristics_matrix[heuristics_matrix > 1] = 1
    return heuristics_matrix
