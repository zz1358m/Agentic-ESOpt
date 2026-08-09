# source: /home/zhi/Agentic-ESOpt/cache/active_runs/aco_bpp_train_sample_t2000_aco_bpp_sample_t2000_from_rep2_20260718_145101/results/pops_best/population_generation_100.json
# method: sample, prefix=2000, batch_size=20
# task: aco_bpp, rep: 2
# train_objective: 202.4

def heuristics_v2(demand, capacity):
    n = len(demand)
    # Sort the items in decreasing order of their sizes
    sorted_demand = sorted(enumerate(demand), key=lambda x: x[1], reverse=True)
    
    # Initialize the number of bins and the heuristics matrix
    bins = [[] for _ in range(capacity // 10 + 1)]  # Assuming the maximum number of items in a bin is 10
    heuristics = [[0] * n for _ in range(n)]
    
    # Initialize the bin count for each item
    bin_count = [0] * n
    
    # Iterate through each item
    for i, size in sorted_demand:
        # Initialize the index of the bin to be added
        bin_index = 0
        # Try to fit the item into the first bin where it fits
        while bin_index < len(bins):
            # Calculate the remaining capacity of the current bin
            remaining_capacity = capacity - sum(demand[j] for j in bins[bin_index])
            if remaining_capacity >= size:
                # If the item fits, add it to the bin and update the heuristics matrix
                bins[bin_index].append(i)
                heuristics[i][i] = 1
                for j in bins[bin_index]:
                    if j!= i:
                        heuristics[i][j] = 1
                        heuristics[j][i] = 1
                bin_count[i] = len(bins[bin_index])
                break
            else:
                bin_index += 1
        else:
            # If the item does not fit in any bin, create a new bin
            bins.append([i])
            heuristics[i][i] = 1
            for j in bins[-1]:
                if j!= i:
                    heuristics[i][j] = 1
                    heuristics[j][i] = 1
            bin_count[i] = len(bins[-1])
    
    # Return the heuristics matrix
    return heuristics
