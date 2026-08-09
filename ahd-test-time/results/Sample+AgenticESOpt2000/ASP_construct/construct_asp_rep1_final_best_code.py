# source: /home/zhi/Agentic-ESOpt/cache/active_runs/construct_asp_train_sample_es_pop20_gen100_sigma0.001_alpha0.0005_construct_asp_sample_es_current_cosine_t2000_rep1_queue_a_gpu0_3_20260720_030717/results/pops_best/population_generation_100.json
# method: sample_es, invalid_reward=current, sigma_schedule=cosine
# population=20, generations=100, samples=2000, sigma=0.001->0, alpha=0.0005
# task: construct_asp, rep: 1
# train_objective: -2769.0

def priority(el, n, w):
    """
    Compute priority of 'el' with respect to the admissible set problem.
    
    Parameters
    ----------
    el : tuple of int
        Vector with elements from {0, 1, 2}.
    n : int
        The length of the vector.
    w : int
        The number of non-zero elements in the vector.
    
    Returns
    -------
    priority : float
        The priority of the vector 'el'.
    """

    # Sort the vector 'el' by the occurrence of its coordinates
    el_sorted = sorted(enumerate(el), key=lambda x: el.count(x[0]), reverse=True)

    # Prioritize the vector based on the occurrence of its coordinates
    priority = 0
    for i in range(n):
        if i < len(el_sorted) and el_sorted[i][1] == 0:
            continue
        elif i > 0 and el_sorted[i][1] == el_sorted[i-1][1]:
            continue
        elif i < len(el_sorted) and el_sorted[i][1] == 2:
            priority += 1
        elif i < len(el_sorted) and el_sorted[i][1] == 3:
            priority += 2
        elif i < len(el_sorted) and el_sorted[i][1] == 4:
            priority += 3
        elif i < len(el_sorted) and el_sorted[i][1] == 5:
            priority += 4
        else:
            priority += 5
    return float(priority)
