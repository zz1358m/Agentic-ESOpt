# source: /home/zhi/Dynamic-Agent/cache/active_runs/aco_bpp_train_eoh_aco_bpp_train_eoh_rep1_rerun_eoh_k1_bpp_3rep_8gpu_20260716_120821/results/pops_best/population_generation_25.json
# run_id: aco_bpp_train_eoh_rep1_rerun_eoh_k1_bpp_3rep_8gpu_20260716_120821
# train_objective: 202.2
# m1m2_multiplier: 1

import numpy as np

def heuristics_v2(demand, capacity):
    n = len(demand)
    heuristics_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if demand[i]!= demand[j]:
                heuristics_matrix[i, j] = (2 * demand[j] * demand[i]) / (demand[i] + demand[j] + (demand[i] - demand[j])**2 / capacity)
            else:
                heuristics_matrix[i, j] = 1
    
    return heuristics_matrix
