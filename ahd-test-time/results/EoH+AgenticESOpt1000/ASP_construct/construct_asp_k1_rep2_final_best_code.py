# source: /home/zhi/Agentic-ESOpt/cache/active_runs/construct_asp_train_es_sigma0.001_alpha0.0005_construct_asp_train_es_full_reload_sigma1e-3_alpha5e-4_rep2_rerun_agentic_esopt_k1_kp_asp_3rep_8gpu_20260716_120821/results/pops_best/population_generation_25.json
# run_id: construct_asp_train_es_full_reload_sigma1e-3_alpha5e-4_rep2_rerun_agentic_esopt_k1_kp_asp_3rep_8gpu_20260716_120821
# train_objective: -2784.0
# m1m2_multiplier: 1.0

import numpy as np

def priority(el, n, w):
    def clump_penalty(el):
        num_clumps = 0
        for i in range(len(el) - 3):
            if el[i] == el[i+1] == el[i+2] == el[i+3] and el[i]!= 2:
                num_clumps += 1
        return num_clumps / (1 + len(el))

    def consecutive_penalty(el):
        num_consecutive = 0
        for i in range(len(el) - 1):
            if el[i] == el[i+1] and el[i]!= 0 and el[i]!= 2:
                num_consecutive += 1
        return num_consecutive / (1 + w)

    def uniformity_bonus(el):
        return 1 / (1 + np.std(el))

    def sparse_penalty(el):
        num_zeros = n - sum(el)
        return num_zeros / (1 + w)

    uni_wt = 0.35
    con_wt = 0.25
    clu_wt = 0.15
    alt_wt = 0.05
    alt_con_wt = 0.05

    spread_bonus = sum(1 for i in range(len(el) - 1) if el[i]!= el[i+1])
    alternation_bonus = sum(1 for i in range(len(el) - 1) if (el[i]!= 0 and el[i+1] == 0) or (el[i] == 0 and el[i+1]!= 0))

    alt_con_bonus = max(alt_con_wt * (len(el) - 1) - consecutive_penalty(el), 0)
    con_alt_bonus = max(alt_con_wt * (len(el) - 1) - consecutive_penalty(el), 0)

    return (uni_wt * uniformity_bonus(el) + alt_wt * alternation_bonus + con_wt * spread_bonus - clu_wt * clump_penalty(el) - con_wt * consecutive_penalty(el) + alt_con_wt * alt_con_bonus + 0.2 * sparse_penalty(el) + 2 * max(alt_con_wt * (len(el) - 1) - consecutive_penalty(el), 0))
