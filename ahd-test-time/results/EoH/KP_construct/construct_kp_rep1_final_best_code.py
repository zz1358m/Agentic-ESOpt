import numpy as np

def select_next_item(remaining_capacity, weights, values, prev_item=None):
    if prev_item is None or weights[prev_item] == 0:
        best_item = -1
        max_value_ratio = 0
        for i in range(len(weights)):
            if weights[i] <= remaining_capacity:
                weight_fraction = weights[i] / (weights[i] + remaining_capacity)
                effective_value_ratio = (values[i] / weights[i]) * (1 + weight_fraction / 2)
                if effective_value_ratio > max_value_ratio:
                    max_value_ratio = effective_value_ratio
                    best_item = i
    else:
        # calculate the pivot value based on the weight of the previously selected item
        pivot = remaining_capacity - weights[prev_item]
        best_item = -1
        max_value_ratio = 0
        for i in range(len(weights)):
            if weights[i] <= remaining_capacity and weights[i] <= pivot:
                weight_fraction = weights[i] / pivot
                effective_value_ratio = (values[i] / weights[i]) * (1 + weight_fraction)
                if effective_value_ratio > max_value_ratio:
                    max_value_ratio = effective_value_ratio
                    best_item = i
    return best_item
