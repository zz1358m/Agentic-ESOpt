# source: /home/zhi/Dynamic-Agent/cache/active_runs/aco_bpp_train_sample_es_pop20_gen100_sigma0.001_alpha0.0005_aco_bpp_sample_es_current_cosine_t2000_rep2_queue_a_gpu0_3_20260720_030717/results/pops_best/population_generation_100.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=100, samples=2000, sigma=0.001->0, alpha=0.0005
# task: aco_bpp, rep: 2
# train_objective: 202.4

# evaluation_repairs: add_missing_numpy_import

import numpy as np

def heuristics_v2(demand, capacity):
    """
    This algorithm first sorts items by size and then assigns them to bins in decreasing size order, 
    giving priority to bins with the most recently opened bin, thus aiming to balance the load in each bin.

    :param demand: A 1D array of the demand sizes of the items.
    :param capacity: An integer representing the capacity of each bin.
    :return: A 2D array where heuristics[i][j] represents how promising it is to put item i and item j in the same bin.
    """

    n = demand.shape[0]
    heuristics = np.zeros((n, n))

    # Create a copy of demand array
    demand_copy = demand.copy()

    # Sort items by size in descending order
    sorted_indices = np.argsort(demand_copy)[::-1]

    bins = []
    for index in sorted_indices:
        item_size = demand_copy[index]
        # Open a new bin if the current item does not fit in any existing bin
        new_bin = True
        for i in range(len(bins)):
            bin_content = sum(demand[bins[i]])
            if bin_content + item_size <= capacity:
                bins[i].append(index)
                heuristics[index, bins[i]] = heuristics[bins[i], index] = 1
                new_bin = False
                break

        if new_bin:
            bins.append([index])
            heuristics[index, bins[-1]] = heuristics[bins[-1], index] = 1

    return heuristics
