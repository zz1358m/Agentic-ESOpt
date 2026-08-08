import numpy as np

def select_next_item(remaining_capacity, weights, values):
    if len(weights) == 0:
        return -1

    # propagate the remaining capacity to items with higher value-to-weight ratios
    weights_relaxed = np.clip(weights + remaining_capacity, 0, np.inf)
    # calculate the weighted average of value and relaxed weight
    item_scores = (values / (weights + 1e-6) * 0.8) + (values / weights_relaxed * 0.2)
    
    # select the item with the highest score
    next_item_idx = np.argmax(item_scores)

    # if the selected item cannot fit within the remaining capacity, relax its weight further
    if weights[next_item_idx] > remaining_capacity:
        # initialize a sorted list of items based on their scores
        sorted_item_idx = np.argsort(-item_scores)
        for item_idx in sorted_item_idx:
            # check if the selected item fits within the remaining capacity
            if weights[item_idx] <= remaining_capacity:
                next_item_idx = item_idx
                break

    return next_item_idx
