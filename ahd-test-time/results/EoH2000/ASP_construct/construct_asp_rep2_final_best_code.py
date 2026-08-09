# Archived from /home/zhi/Agentic-ESOpt/cache/active_runs/construct_asp_train_eoh_construct_asp_train_eoh_rep2_orig_eoh_all6_k3_8gpu_20260713_142341/results/pops_best/population_generation_25.json
# run_id: construct_asp_train_eoh_rep2_orig_eoh_all6_k3_8gpu_20260713_142341
# train_objective: -2763.0
# method: original EoH, population=10, generations=25, k=3 replicates

def priority(el, n, w):
    count = len([i for i in el if i!= 0])
    if count == 0:
        return 0
    divergence = sum(abs(el[i] - el[i-1]) for i in range(1, n))
    non_zero_diffs = [abs(el[i] - el[(i+1) % n]) for i in range(n) if el[i]!= 0]
    divergence_div = (divergence / count) * (min(n, count) / max(1, count - 1))
    non_zero_diffs_squared = sum(i ** 2 for i in non_zero_diffs)
    non_zero_ratio = count / w
    max_unique_positions = min(n, w)
    return non_zero_ratio * (divergence_div + non_zero_diffs_squared / max_unique_positions)
