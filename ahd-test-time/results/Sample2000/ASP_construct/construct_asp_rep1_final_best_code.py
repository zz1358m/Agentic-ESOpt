# source: /home/zhi/Dynamic-Agent/cache/active_runs/construct_asp_train_sample_t2000_construct_asp_sample_t2000_from_rep1_20260718_145101/results/pops_best/population_generation_100.json
# method: sample, prefix=2000, batch_size=20
# task: construct_asp, rep: 1
# train_objective: -2754.0

def priority(el, n, w):
    runs = 0
    last_non_zero = -1
    for i in range(n):
        if el[i]!= 0:
            if i!= last_non_zero + 1:
                runs += 1
            last_non_zero = i
    max_val = max(el)
    min_val = min(el)
    return (max_val - min_val) + runs - (max_val - 1) * (min_val == 0) - (n - (max_val > 2))
