# source: /home/zhi/Agentic-ESOpt/cache/active_runs/construct_asp_train_sample_es_pop20_gen100_sigma0.001_alpha0.0005_construct_asp_sample_es_current_cosine_t2000_rep2_queue_b_gpu4_7_20260720_030717/results/pops_best/population_generation_100.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=100, samples=2000, sigma=0.001->0, alpha=0.0005
# task: construct_asp, rep: 2
# train_objective: -2754.0

def priority(el, n, w):
    count_00x = sum(1 for i in range(n) if el[i] == 0 and (i + 1 in [j for j in range(n) if el[j]!= 0]))
    count_zero = sum(1 for i in range(n) if el[i] == 0)
    count_unique = len(set(i for i in range(n) if el[i]!= 0))
    return -count_zero - count_unique + count_00x
