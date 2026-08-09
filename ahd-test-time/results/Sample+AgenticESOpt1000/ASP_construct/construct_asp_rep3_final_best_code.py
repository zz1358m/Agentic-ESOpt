# source: /home/zhi/Agentic-ESOpt/cache/active_runs/construct_asp_train_sample_es_pop20_gen50_sigma0.001_alpha0.0005_construct_asp_sample_es_current_cosine_t1000_rep3_queue_a_gpu0_3_20260720_030717/results/pops_best/population_generation_50.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=50, samples=1000, sigma=0.001->0, alpha=0.0005
# task: construct_asp, rep: 3
# train_objective: -2703.0

def priority(el, n, w):
    count = 0
    for i in range(1, n):
        count += max(1 - el[i] * el[i - 1], 0)
    return count / (w * w / 2)
