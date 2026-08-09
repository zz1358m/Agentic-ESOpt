# source: /home/zhi/Agentic-ESOpt/cache/active_runs/aco_bpp_train_eoh_aco_bpp_train_eoh_rep2_rerun_eoh_k1_bpp_3rep_8gpu_20260716_120821/results/pops_best/population_generation_25.json
# run_id: aco_bpp_train_eoh_rep2_rerun_eoh_k1_bpp_3rep_8gpu_20260716_120821
# train_objective: 202.4
# m1m2_multiplier: 1

import numpy as np

def heuristics_v2(demand, capacity):
    n = len(demand)
    heuristics = np.zeros((n, n))

    demand_sorted = np.argsort(demand)[::-1]
    bins = [[] for _ in range(n)]
    occupied_bins = 0
    bin_load = [0] * n

    for i in demand_sorted:
        for j in range(occupied_bins):
            if bin_load[j] + demand[i] <= capacity:
                bin_load[j] += demand[i]
                bins[j].append(i)
                break
        else:
            bin_load[occupied_bins] += demand[i]
            bins[occupied_bins].append(i)
            occupied_bins += 1

    for bin_idx in range(occupied_bins):
        for j in bins[bin_idx]:
            for k in bins[bin_idx]:
                heuristics[j, k] = 1 / (bin_load[bin_idx] / capacity + 1e-9)

    return heuristics
