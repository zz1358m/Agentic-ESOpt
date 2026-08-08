# Archived from /home/zhi/Dynamic-Agent/cache/active_runs/aco_bpp_train_eoh_aco_bpp_train_eoh_rep2_unfinished_true8_cvrp_bpp_20260715_053235/results/pops_best/population_generation_25.json
# run_id: aco_bpp_train_eoh_rep2_unfinished_true8_cvrp_bpp_20260715_053235
# train_objective: 202.4
# method: original EoH, population=10, generations=25, k=3 replicates

def heuristics_v3(demand, capacity):
    n = len(demand)
    heuristics_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    total_demand = sum(demand)
    for i in range(n):
        for j in range(i+1, n):
            combined_demand = demand[i] + demand[j]
            if demand[i] == 0 or demand[j] == 0 or combined_demand > capacity:
                heuristics_matrix[i][j] = heuristics_matrix[j][i] = 0
            else:
                similarity_score = (demand[i] - demand[j]) ** 2 / (demand[i] ** 2 + demand[j] ** 2) if demand[i]!= demand[j] else 1.0
                score = (combined_demand / total_demand) * (1 - similarity_score)
                heuristics_matrix[i][j] = heuristics_matrix[j][i] = score
    return heuristics_matrix
