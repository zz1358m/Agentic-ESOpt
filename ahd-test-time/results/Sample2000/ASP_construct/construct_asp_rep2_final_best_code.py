# source: /home/zhi/Agentic-ESOpt/cache/active_runs/construct_asp_train_sample_t2000_construct_asp_sample_t2000_from_rep2_20260718_145101/results/pops_best/population_generation_100.json
# method: sample, prefix=2000, batch_size=20
# task: construct_asp, rep: 2
# train_objective: -2769.0

def priority(el, n, w):
    score = 0
    consecutive_zeros = 1
    for i in range(1, n):
        if el[i] == 0 and el[i-1] == 0:
            consecutive_zeros += 1
        else:
            if consecutive_zeros > 1:
                score -= consecutive_zeros ** 2
            consecutive_zeros = 1
    if consecutive_zeros > 1:
        score -= consecutive_zeros ** 2

    unique_positions = len([x for x in set(el) if x!= 0])
    score += unique_positions * 2 - w

    score += 10 * (2 <= sum(el) <= n)

    score += 5 * sum(abs(el[i] - el[i-1]) > 0 for i in range(1, n)) if len(set(el)) == 3 else 0

    return score / max(n, 1)
