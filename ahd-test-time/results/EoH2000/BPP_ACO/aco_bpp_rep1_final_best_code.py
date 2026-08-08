# Archived from /home/zhi/Dynamic-Agent/cache/active_runs/aco_bpp_train_eoh_aco_bpp_train_eoh_rep1_unfinished_true8_cvrp_bpp_20260715_053235/results/pops_best/population_generation_25.json
# run_id: aco_bpp_train_eoh_rep1_unfinished_true8_cvrp_bpp_20260715_053235
# train_objective: 202.4
# method: original EoH, population=10, generations=25, k=3 replicates

def heuristics_v3(demand, capacity):
    n = len(demand)
    demand_sorted = sorted(enumerate(demand), key=lambda x: x[1], reverse=True)
    bins = [[] for _ in range(n)]
    assignments = [[] for _ in range(n)]

    for i, d in demand_sorted:
        bin_id = 0
        while bin_id < len(bins) and sum(demand[j] for j in bins[bin_id]) + d > capacity:
            bin_id += 1
        if bin_id < len(bins):
            bins[bin_id].append(i)
            assignments[bin_id].append(i)
        else:
            bin_id = len(bins)
            bins.append([i])
            assignments.append([i])

    heuristics_matrix = [[0 for _ in range(n)] for _ in range(n)]
    for bin_id in range(len(bins)):
        for i in assignments[bin_id]:
            for j in assignments[bin_id]:
                if i!= j:
                    heuristics_matrix[i][j] = heuristics_matrix[j][i] = 1

    for i in range(n):
        max_heuristic = 0
        for j in range(n):
            if i!= j:
                max_heuristic = max(max_heuristic, heuristics_matrix[j][i])
        heuristics_matrix[i][i] = max_heuristic

    return heuristics_matrix
