# source: /home/zhi/Agentic-ESOpt/cache/active_runs/aco_bpp_train_sample_es_pop20_gen50_sigma0.001_alpha0.0005_aco_bpp_sample_es_current_cosine_t1000_rep3_queue_b_gpu4_7_20260720_030717/results/pops_best/population_generation_50.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=50, samples=1000, sigma=0.001->0, alpha=0.0005
# task: aco_bpp, rep: 3
# train_objective: 202.4

def heuristics_v2(demand, capacity):
    """
    The First-Fit Decreasing algorithm's heuristics matrix.

    The algorithm works by placing each bin in descending order of demand.
    For each bin, it tries to find the best items to pack in this bin.

    The heuristics for an item pair (i, j) is calculated based on their order in the bins they are packed in.
    If items i and j are packed in the same bin, their heuristics is 1 (promising).
    If one item is packed in bin k and the other is packed in bin l (k!= l), the heuristics is the proportion of the 
    smaller bin's capacity to the capacity of bin k (for item i) and the proportion of the smaller bin's capacity to the capacity of 
    bin l (for item j), then their product.
    If both items are not packed in the same bins, their heuristics is 0 (not promising).
    """
    n = len(demand)
    bin_sizes = [capacity] * n  # Initialize all bins with full capacity
    bin_contents = [[] for _ in range(n)]  # Initialize all bins as empty
    sorted_demand = sorted(enumerate(demand), key=lambda x: x[1], reverse=True)  # Sort items in descending order of demand
    heuristics_matrix = [[0] * n for _ in range(n)]

    for i, (item_index, demand_size) in enumerate(sorted_demand):
        placed = False
        for j in range(n):
            # Check if the demand of the current item can be packed in the current bin
            if bin_sizes[j] >= demand_size:
                # Add the current item to the current bin
                bin_contents[j].append(item_index)
                bin_sizes[j] -= demand_size  # Update the remaining capacity of the bin
                heuristics_matrix[item_index][item_index] = 1  # Update heuristics for the pair (i, i) to 1
                placed = True  # Mark that the item has been placed
                break
        if not placed:
            raise ValueError(f"Unpacking all items was not possible due to capacity constraints")

        # For each other item in the bin with the current item, update their heuristics
        for other_item in bin_contents[j]:
            if other_item!= item_index:
                # Calculate the proportion of the smaller bin's capacity to the capacity of bin k (for item i)
                # and the proportion of the smaller bin's capacity to the capacity of bin l (for item j)
                heuristics_matrix[item_index][other_item] = (demand_size / (bin_sizes[j] + demand_size)) * (demand_size / (bin_sizes[j] + demand_size))
                heuristics_matrix[other_item][item_index] = heuristics_matrix[item_index][other_item]

    return heuristics_matrix
