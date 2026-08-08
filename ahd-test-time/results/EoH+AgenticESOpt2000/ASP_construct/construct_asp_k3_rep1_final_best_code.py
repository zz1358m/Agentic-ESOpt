# Archived from /home/zhi/Dynamic-Agent/cache/active_runs/construct_asp_train_es_sigma0.001_alpha0.0005_construct_asp_train_es_full_reload_sigma1e-3_alpha5e-4_rep1_k3_decay_full_all6_3rep_reload8_20260705_121329/results/pops_best/population_generation_25.json
# run_id: construct_asp_train_es_full_reload_sigma1e-3_alpha5e-4_rep1_k3_decay_full_all6_3rep_reload8_20260705_121329
# train_objective: -2796.0
# m1m2_multiplier: 3.0
# sigma_schedule: cosine
# final_model_es_sigma: 6.698729810778065e-05

def priority(el, n, w):
    def longest_run(el, x):
        max_run = 0
        current_run = 0
        for i in range(n):
            if el[i] == x:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        return max_run

    def num_unique_nonzeros(el):
        return len(set(x for x in el if x!= 0))

    def balanced_distribution(el):
        balanced = 1 / (1 + sum(longest_run(el, x) for x in {0, 1}))
        return balanced

    def disperse_unique_positions(el):
        unique_pos = 0
        for i in range(n):
            if i > 0 and el[i-1]!= el[i]:
                unique_pos += 1
        return unique_pos / n

    def frequency_1(el):
        return (len([x for x in el if x == 1]) / n) ** 2

    return 0.5 * (balanced_distribution(el) * 2 ** (-1 * longest_run(el, 0))) + 0.3 * disperse_unique_positions(el) + 0.2 * frequency_1(el)
