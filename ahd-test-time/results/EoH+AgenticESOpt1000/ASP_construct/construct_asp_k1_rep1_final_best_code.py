# source: /home/zhi/Dynamic-Agent/cache/active_runs/construct_asp_train_es_sigma0.001_alpha0.0005_construct_asp_train_es_full_reload_sigma1e-3_alpha5e-4_rep1_rerun_dynamic_k1_kp_asp_3rep_8gpu_20260716_120821/results/pops_best/population_generation_25.json
# run_id: construct_asp_train_es_full_reload_sigma1e-3_alpha5e-4_rep1_rerun_dynamic_k1_kp_asp_3rep_8gpu_20260716_120821
# train_objective: -2766.0
# m1m2_multiplier: 1.0

def priority(el, n, w):
    variance = sum([(x - sum(el) / len(el)) ** 2 for x in el]) / len(el)
    num_unique_elements = len(set(el))
    zeros = el.count(0)
    blocks = 0
    previous_element = None
    for i in range(len(el)):
        if el[i] == 0 and previous_element == 0:
            blocks += 1
        previous_element = el[i]
    return (variance / (zeros / n) + num_unique_elements - blocks) / (sum(el) / len(el) + 1)
