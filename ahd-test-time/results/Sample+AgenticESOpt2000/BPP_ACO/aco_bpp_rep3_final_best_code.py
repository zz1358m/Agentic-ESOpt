# source: /home/zhi/Agentic-ESOpt/cache/active_runs/aco_bpp_train_sample_es_pop20_gen100_sigma0.001_alpha0.0005_aco_bpp_sample_es_current_cosine_t2000_rep3_queue_b_gpu4_7_20260720_030717/results/pops_best/population_generation_100.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=100, samples=2000, sigma=0.001->0, alpha=0.0005
# task: aco_bpp, rep: 3
# train_objective: 202.4

def heuristics_v2(demand, capacity):
    """
    {The heuristics_v2 function implements the Next Fit Decreasing algorithm to solve the Bin Packing Problem. 
    It iterates over the items in decreasing order of size, placing each item into the first bin that has available space. 
    The function returns a heuristics matrix where each element heuristics[i][j] represents how promising it is to put item i and item j in the same bin.}
    """
    
    # Sort the items in decreasing order of size
    sorted_items = sorted(range(len(demand)), key=lambda i: demand[i], reverse=True)
    
    # Initialize the bins with the given capacity
    n = len(demand)
    bins = [[] for _ in range(n)]
    bin_allocations = [0] * n
    
    # Initialize the heuristics matrix
    heuristics = [[0] * n for _ in range(n)]
    
    # Iterate over the sorted items
    for i in sorted_items:
        # Initialize a variable to store the best bin for the current item
        best_bin = None
        
        # Iterate over the existing bins
        for j, bin in enumerate(bins):
            # Check if the current item can be placed in the bin
            if sum(demand[k] for k in bin) + demand[i] <= capacity:
                best_bin = j
                break
        
        # If no bin was found, create a new one
        if best_bin is None:
            best_bin = len(bins)
            bins.append([])
        
        # Add the item to the bin
        bins[best_bin].append(i)
        bin_allocations[i] = best_bin
    
    # Update the heuristics matrix
    for i in range(n):
        for j in range(n):
            heuristics[i][j] = 1 if bin_allocations[i] == bin_allocations[j] else 0
    
    return heuristics
