# source: /home/zhi/Agentic-ESOpt/cache/active_runs/construct_asp_train_sample_es_pop20_gen50_sigma0.001_alpha0.0005_construct_asp_sample_es_current_cosine_t1000_rep1_queue_b_gpu4_7_20260720_030717/results/pops_best/population_generation_50.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=50, samples=1000, sigma=0.001->0, alpha=0.0005
# task: construct_asp, rep: 1
# train_objective: -2754.0

def priority(el, n, w):
    priority = 0
    cnt_zero = [0] * n
    cnt_non_zero = [0] * n
    for i in range(n):
        if el[i]!= 0:
            cnt_non_zero[i] = 1
        else:
            cnt_zero[i] = 1
    for i in range(n - 1):
        if el[i] == 0 and el[i+1]!= 0:
            cnt_non_zero[i+1] += 1
            priority += 2
    for i in range(n):
        if el[i]!= 0:
            priority -= cnt_zero[i]
    if w < n:
        priority += w * (w - 1) / (2 * n)
    return priority
