# source: /home/zhi/Agentic-ESOpt/cache/active_runs/aco_bpp_train_sample_t1000_aco_bpp_sample_t1000_rep3_20260718_144918/results/pops_best/population_generation_50.json
# method: sample, prefix=1000, batch_size=20
# task: aco_bpp, rep: 3
# train_objective: 202.4

def heuristics_v2(demand, capacity):
    {
        # The heuristics algorithm is based on the First-Fit Decreasing (FFD) algorithm, 
        # where it first sorts the items in decreasing order of their sizes, 
        # then iterates over the items, trying to pack each one in the first bin where it fits, 
        # and finally calculates the promisingness of every pair of items based on the bins they are packed into.
    }

    n = len(demand)
    demand_sorted = sorted(range(n), key=lambda i: demand[i], reverse=True)
    bins = []
    heuristics = [[0]*n for _ in range(n)]

    for item in demand_sorted:
        placed = False
        for bin in bins:
            if sum(demand[i] for i in bin) + demand[item] <= capacity:
                bin.append(item)
                heuristics[item][item] = 1
                for p in bin:
                    if p!= item:
                        heuristics[item][p] = 1
                for other_item, other_bin in enumerate(bins):
                    if other_item!= bin:
                        intersection = set(bin) & set(other_bin)
                        if intersection:
                            for i in bin:
                                for j in other_bin:
                                    if i!= j:
                                        heuristics[i][j] = 1
                                    else:
                                        heuristics[i][j] = 2
                            break
                placed = True
                break
        if not placed:
            bins.append([item])
            heuristics[item][item] = 1

    return heuristics
