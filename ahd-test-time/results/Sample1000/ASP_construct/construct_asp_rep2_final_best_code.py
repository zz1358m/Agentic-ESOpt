# source: /home/zhi/Agentic-ESOpt/cache/active_runs/construct_asp_train_sample_t1000_construct_asp_sample_t1000_rep2_20260718_144918/results/pops_best/population_generation_50.json
# method: sample, prefix=1000, batch_size=20
# task: construct_asp, rep: 2
# train_objective: -2751.0

def priority(el, n, w):
    if len(el)!= n:
        raise ValueError("Input 'el' must be of length n")

    count_0, count_1, count_2 = el.count(0), el.count(1), el.count(2)
    
    repetition_penalty = 0
    unique_penalty = 0
    
    for i in range(n):
        if el[i] == el[i-1] and i > 0:
            repetition_penalty += 1
        if el[i] == el[(i-1)%n] and el[(i+1)%n] == el[i]: # also consider circular case
            repetition_penalty += 1
    
    unique_index = [i for i, x in enumerate(el) if x == 0 or x == 2]
    unique_penalty = len(unique_index)
    
    return 1 / ((repetition_penalty + 1) * (unique_penalty + 1) + 1)
