# Archived from /home/zhi/Dynamic-Agent/cache/active_runs/aco_bpp_train_eoh_aco_bpp_train_eoh_rep3_unfinished_true8_cvrp_bpp_20260715_053235/results/pops_best/population_generation_25.json
# run_id: aco_bpp_train_eoh_rep3_unfinished_true8_cvrp_bpp_20260715_053235
# train_objective: 202.6
# method: original EoH, population=10, generations=25, k=3 replicates

import numpy as np

def heuristics_v2(demand, capacity):
    n = len(demand)
    total_volume = demand.sum() - demand.max()
    
    heuristics = np.zeros((n, n), dtype=float)
    
    for i in range(n):
        for j in range(n):
            if j!= i:
                volume = np.min([demand[i], demand[j]])
                heuristics[i, j] = heuristics[j, i] = (volume / capacity) + (np.abs(demand[i] - demand[j]) / total_volume) * (demand[i] / demand.max())

    return heuristics
