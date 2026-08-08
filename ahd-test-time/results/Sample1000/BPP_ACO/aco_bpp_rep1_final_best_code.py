# source: /home/zhi/Dynamic-Agent/cache/active_runs/aco_bpp_train_sample_t1000_aco_bpp_sample_t1000_rep1_20260718_091608/results/pops_best/population_generation_50.json
# method: sample, prefix=1000, batch_size=20
# task: aco_bpp, rep: 1
# train_objective: 202.4

def heuristics_v2(demand, capacity):
    # Sort the items in decreasing order of their sizes
    sorted_demand = sorted(range(len(demand)), key=lambda i: demand[i], reverse=True)
    
    # Initialize the bins
    bins = []
    
    # Initialize the heuristics matrix
    n = len(demand)
    heuristics = [[0]*n for _ in range(n)]
    
    # Iterate through each item
    for i in sorted_demand:
        # Initialize a flag to check if the item is placed in a bin
        placed = False
        
        # Iterate through each bin
        for j in range(len(bins)):
            # Check if the item fits in the current bin
            if demand[i] + (sum([demand[k] for k in bins[j]]) if bins[j] else 0) <= capacity:
                # Update the heuristics matrix
                for k in bins[j]:
                    heuristics[i][k] += 1
                    heuristics[k][i] += 1
                
                # Add the item to the bin
                bins[j].append(i)
                placed = True
                break
        
        # If the item cannot be placed in any existing bin, create a new bin
        if not placed:
            bins.append([i])
    
    return heuristics
