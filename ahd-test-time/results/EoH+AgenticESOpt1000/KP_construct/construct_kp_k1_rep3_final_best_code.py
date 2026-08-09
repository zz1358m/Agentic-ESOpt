# source: /home/zhi/Agentic-ESOpt/cache/active_runs/construct_kp_train_es_sigma0.001_alpha0.0005_construct_kp_train_es_full_reload_sigma1e-3_alpha5e-4_rep3_rerun_agentic_esopt_k1_kp_asp_3rep_8gpu_20260716_120821/results/pops_best/population_generation_25.json
# run_id: construct_kp_train_es_full_reload_sigma1e-3_alpha5e-4_rep3_rerun_agentic_esopt_k1_kp_asp_3rep_8gpu_20260716_120821
# train_objective: -40.15958
# m1m2_multiplier: 1.0

import numpy as np

def select_next_item(remaining_capacity, weights, values):
    fit_indices = np.where(weights <= remaining_capacity)[0]
    
    if len(fit_indices) == 0:
        return -1
    
    ratios = values[fit_indices] / (weights[fit_indices] + remaining_capacity / (len(fit_indices) + 1))
    
    next_item_index = fit_indices[np.argmax(ratios)]
    
    return next_item_index
