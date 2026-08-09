# Archived from /home/zhi/Agentic-ESOpt/cache/active_runs/aco_bpp_train_es_sigma0.001_alpha0.0005_aco_bpp_train_es_full_reload_sigma1e-3_alpha5e-4_rep3_k3_decay_full_all6_3rep_reload8_20260705_121329/results/pops_best/population_generation_25.json
# run_id: aco_bpp_train_es_full_reload_sigma1e-3_alpha5e-4_rep3_k3_decay_full_all6_3rep_reload8_20260705_121329
# train_objective: 202.4
# m1m2_multiplier: 3.0
# sigma_schedule: cosine
# final_model_es_sigma: 6.698729810778065e-05

def heuristics_v2(demand, capacity):
    n = len(demand)
    heuristics = [[1 - abs((demand[i] + demand[j] - capacity) / (demand[i] + demand[j])) for j in range(n)] for i in range(n)]

    # Sort the items by demand in descending order
    indices = list(range(n))
    demand_copy = [x for x in demand]
    indices.sort(key=lambda i: demand[i], reverse=True)

    for i in range(n):
        if heuristics[i][i] >= 0.5:
            # Select the items with the largest demand first and pack them in the bin where the current item is packed
            for j in range(n):
                if demand[j] > demand[i] and heuristics[i][j] < 1:
                    heuristics[i][j] = heuristics[j][i] = 1 - abs((demand[i] + demand[j] - capacity) / (demand[i] + demand[j]))
        else:
            # Also, pack the items with low demand into the bin if it has remaining capacity
            remaining_capacity = capacity
            for j in indices:
                if j!= i and demand[j] < demand[i] and remaining_capacity >= demand[j]:
                    heuristics[i][j] = heuristics[j][i] = 1
                    remaining_capacity -= demand[j]

    return heuristics
