# Archived from /home/zhi/Agentic-ESOpt/cache/active_runs/construct_asp_train_eoh_construct_asp_train_eoh_rep3_orig_eoh_all6_k3_8gpu_20260713_142341/results/pops_best/population_generation_25.json
# run_id: construct_asp_train_eoh_rep3_orig_eoh_all6_k3_8gpu_20260713_142341
# train_objective: -2775.0
# method: original EoH, population=10, generations=25, k=3 replicates

def priority(el, n, w):
    def count_consecutive_max(el):
        max_consecutive = 0
        current_consecutive = 1
        for i in range(1, n):
            if el[i] == el[i - 1] == 0:
                current_consecutive += 1
            else:
                max_consecutive = max(max_consecutive, current_consecutive)
                current_consecutive = 1
        return max(max_consecutive, current_consecutive)

    def count_consecutive_non_zero(el):
        max_consecutive = 0
        current_consecutive = 1
        for i in range(1, n):
            if el[i] > 0 and el[i - 1] > 0:
                current_consecutive += 1
            else:
                max_consecutive = max(max_consecutive, current_consecutive)
                current_consecutive = 1
        return max(max_consecutive, current_consecutive)

    def count_unique_positions(el):
        return len(set(i for i, x in enumerate(el) if x > 0))

    def count_non_zero_values(el):
        return sum(1 for x in el if x > 0)

    def count_non_zero_density(el):
        non_zero_values = count_non_zero_values(el)
        return non_zero_values / n if non_zero_values > 0 else 1

    def count_consecutive_zeros(el):
        max_consecutive = 0
        current_consecutive = 1
        for i in range(1, n):
            if el[i] == 0 and el[i - 1] == 0:
                current_consecutive += 1
            else:
                max_consecutive = max(max_consecutive, current_consecutive)
                current_consecutive = 1
        return max(max_consecutive, current_consecutive)

    uq = count_unique_positions(el) / n
    c = count_consecutive_max(el)
    c_nz = count_consecutive_non_zero(el)
    c_z = count_consecutive_zeros(el)
    non_zero = count_non_zero_density(el)
    u = count_unique_positions(el)
    return u * 1e2 / (c * 1e5 + c_nz * 1e5 + c_z * 1e4 + non_zero * 1e3)
