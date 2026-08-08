# Archived from /home/zhi/Dynamic-Agent/cache/active_runs/construct_asp_train_es_sigma0.001_alpha0.0005_construct_asp_train_es_full_reload_sigma1e-3_alpha5e-4_rep3_k3_decay_full_all6_3rep_reload8_20260705_121329/results/pops_best/population_generation_25.json
# run_id: construct_asp_train_es_full_reload_sigma1e-3_alpha5e-4_rep3_k3_decay_full_all6_3rep_reload8_20260705_121329
# train_objective: -2778.0
# m1m2_multiplier: 3.0
# sigma_schedule: cosine
# final_model_es_sigma: 0.0009330127018922195

def priority(el, n, w):
    if not el or w > len(el):
        return 0.0  # Return float for consistency
    cons = 0
    last_freq = 0
    alt = 0
    curr = el[0]
    last_pos = 0
    for i in range(1, len(el)):
        if el[i] == curr:
            cons += 1
        elif el[i] == el[i-1]:
            if i - last_pos > 1:
                alt += 1
            last_pos = i
        else:
            if el[i] == 1:
                last_freq += 1
            curr = el[i]
    score1 = -cons  # higher priority to vectors with more consecutive equal elements
    score2 = 20 * (last_freq / w) if w > 0 else 0  # higher priority to vectors with more frequent values 1 at the last positions of non-zero elements
    score3 = -alt  # higher priority to vectors with more alternating patterns
    return 30 * score1 + 20 * score2 + score3
