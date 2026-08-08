# Archived from /home/zhi/Dynamic-Agent/cache/active_runs/construct_asp_train_eoh_construct_asp_train_eoh_rep1_orig_eoh_all6_k3_8gpu_20260713_142341/results/pops_best/population_generation_25.json
# run_id: construct_asp_train_eoh_rep1_orig_eoh_all6_k3_8gpu_20260713_142341
# train_objective: -2754.0
# method: original EoH, population=10, generations=25, k=3 replicates

def priority(el, n, w):
    def disjoint_intervals(el):
        if 0 not in el:
            return 0
        intervals = 1
        max_interval = 1
        start = 0
        for i in range(1, len(el)):
            if el[i]!= 0 and el[i-1] == 0:
                start = i
                intervals += 1
                max_interval = max(max_interval, i - start + 1)
            elif el[i]!= 0 and el[i-1]!= 0:
                max_interval = max(max_interval, i - start)
        return intervals, max_interval

    intervals, max_interval = disjoint_intervals(el)
    zero_positions = len(el) - sum(1 for x in el if x!= 0)
    return 0.7 * intervals + 0.3 * (zero_positions / (1 + n))
