# Archived from /home/zhi/Agentic-ESOpt/cache/active_runs/construct_asp_train_es_sigma0.001_alpha0.0005_construct_asp_train_es_full_reload_sigma1e-3_alpha5e-4_rep2_k3_decay_full_all6_3rep_reload8_20260705_121329/results/pops_best/population_generation_25.json
# run_id: construct_asp_train_es_full_reload_sigma1e-3_alpha5e-4_rep2_k3_decay_full_all6_3rep_reload8_20260705_121329
# train_objective: -2778.0
# m1m2_multiplier: 3.0
# sigma_schedule: cosine
# final_model_es_sigma: None

def priority(el, n, w):
    bad_score = 0
    max_run = 1
    max_run_len = 0
    for i in range(1, n):
        if el[i - 1] == el[i] and el[i]!= 2:
            max_run += 1
            max_run_len = max(max_run_len, max_run)
        else:
            bad_score += max_run - 1
            max_run = 1
    bad_score += max_run - 1

    repetitive_sum = 0
    for i in range(n):
        if el[i]!= 2:
            repetitive_sum += el[i]

    good_score = 1 / max_run_len if max_run_len > 1 else 0
    max_non_repetitive = max(el[:w])
    good_score += max_non_repetitive

    if n >= w:
        return -bad_score + repetitive_sum + good_score
    else:
        return -bad_score + repetitive_sum + good_score / 1e9
