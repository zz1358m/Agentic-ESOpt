# source: /home/zhi/Dynamic-Agent/cache/active_runs/aco_bpp_train_sample_es_pop20_gen50_sigma0.001_alpha0.0005_aco_bpp_sample_es_current_cosine_t1000_rep1_queue_b_gpu4_7_20260720_030717/results/pops_best/population_generation_50.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=50, samples=1000, sigma=0.001->0, alpha=0.0005
# task: aco_bpp, rep: 1
# train_objective: 202.4

def heuristics_v2(demand, capacity):
    n = len(demand)
    heuristics = [[0 for _ in range(n)] for _ in range(n)]
    
    bins = []
    
    for i in sorted(range(n), key=lambda x: demand[x], reverse=True):
        assigned = False
        for j, bin in enumerate(bins):
            if sum(demand[k] for k in bin) + demand[i] <= capacity:
                bins[j].append(i)
                for k in bin:
                    heuristics[i][k] += 1
                    heuristics[k][i] += 1
                assigned = True
                break
        if not assigned:
            bins.append([i])
    
    # Adjust suitability of items in the same bin
    for i in range(n):
        bin = next((j for j in range(len(bins)) if i in bins[j]), -1)
        if bin!= -1:
            for j in bins[bin]:
                if i!= j:
                    heuristics[i][j] += 1
                    heuristics[j][i] += 1
    
    return heuristics
