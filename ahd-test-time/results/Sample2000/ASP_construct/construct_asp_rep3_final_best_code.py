# source: /home/zhi/Dynamic-Agent/cache/active_runs/construct_asp_train_sample_t2000_construct_asp_sample_t2000_from_rep3_20260718_145101/results/pops_best/population_generation_100.json
# method: sample, prefix=2000, batch_size=20
# task: construct_asp, rep: 3
# train_objective: -2754.0

def priority(el, n, w):
    el_tuple = list(el)
    priority = 0.0

    # Count the number of unique values
    unique_count = len(set(el_tuple))
    priority += unique_count / n

    # Count the number of pairs with values {0, 1} and {0, 2}
    pairs = 0
    for i in range(n - 1):
        if el_tuple[i] == 0 and el_tuple[i + 1] in [1, 2]:
            pairs += 1
    priority += pairs / n

    # Count the number of consecutive zeros
    zeros = 0
    count = 1
    for i in range(n - 1):
        if el_tuple[i] == 0 and el_tuple[i + 1] == 0:
            count += 1
        else:
            zeros += max(0, count - 1)
            count = 1
    zeros += max(0, count - 1)
    priority -= zeros / n

    return priority
