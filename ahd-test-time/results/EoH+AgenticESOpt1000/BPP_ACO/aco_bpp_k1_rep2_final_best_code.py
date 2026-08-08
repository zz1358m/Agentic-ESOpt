# Archived from /home/zhi/Dynamic-Agent/cache/active_runs/aco_bpp_train_es_sigma0.001_alpha0.0005_aco_bpp_train_es_full_reload_sigma1e-3_alpha5e-4_rep2_k1_decay_full_all6_3rep_reload8_20260706_153046/results/pops_best/population_generation_25.json
# run_id: aco_bpp_train_es_full_reload_sigma1e-3_alpha5e-4_rep2_k1_decay_full_all6_3rep_reload8_20260706_153046
# train_objective: 202.4
# m1m2_multiplier: 1.0
# sigma_schedule: cosine
# final_model_es_sigma: None

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

def heuristics_v2(demand, capacity):
    n = len(demand)
    Z = linkage(demand[:, None])
    clusters = fcluster(Z, 2, criterion='maxclust')
    
    heuristics = np.zeros((n, n))
    
    # assign items in the same cluster to the same bin
    for cluster_id in set(clusters):
        cluster_items = [i for i in range(n) if clusters[i] == cluster_id]
        cluster_demand = demand[cluster_items]
        cluster_bins = []
        
        for item in sorted(cluster_items, key=lambda i: demand[i], reverse=True):
            item_demand = demand[item]
            placed = False
            for bin_idx in range(len(cluster_bins)):
                bin_cap = capacity - sum([demand[j] for j in cluster_bins[bin_idx]])
                if item_demand <= bin_cap:
                    heuristics[item, cluster_bins[bin_idx][0]] = 1
                    heuristics[cluster_bins[bin_idx][0], item] = 1
                    cluster_bins[bin_idx].append(item)
                    placed = True
                    break
            if not placed:
                cluster_bins.append([item])
    
    return heuristics
