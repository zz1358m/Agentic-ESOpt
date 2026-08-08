# source: /home/zhi/Dynamic-Agent/cache/active_runs/construct_asp_train_sample_es_pop20_gen100_sigma0.001_alpha0.0005_construct_asp_sample_es_current_cosine_t2000_rep3_queue_a_gpu0_3_20260720_030717/results/pops_best/population_generation_100.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=100, samples=2000, sigma=0.001->0, alpha=0.0005
# task: construct_asp, rep: 3
# train_objective: -2730.0

def priority(el, n, w):
    """
    {Heuristics for scoring each vector to show the priority in the admissible set:
        1. Weighted sum of the frequency of each coordinate value in el.
        2. Frequency difference between the maximum and minimum frequency of any coordinate value in el.
        3. Frequency difference between the maximum frequency of the most frequent value and the sum of the frequencies of the other two values.
        4. The number of times the same value repeats consecutively in el.
    }
    """
    import collections
    counts = collections.Counter(el)
    counts_v = sum(counts.values())
    freq = {}
    for k in counts:
        if counts_v > 0 and k!= 0:
            freq[k] = counts[k] / counts_v

    priority = 0
    if len(freq) == 3:
        priority = (len(max(freq, key=freq)) + 1) / w / (max(freq.values()) + min(freq.values()))

    for i in range(n - 1):
        window = tuple(el[i:i + 2])
        if list(window) == [0, 0]:
            priority -= 100
        elif len(set(window)) == 2:
            if window.count(window[0])!= 1:
                priority += 1
        elif list(window) == [0, 1]:
            priority += 0.5
        elif list(window) == [0, 2]:
            priority += 1
        elif list(window) == [1, 2]:
            priority += 0.5
    return priority
