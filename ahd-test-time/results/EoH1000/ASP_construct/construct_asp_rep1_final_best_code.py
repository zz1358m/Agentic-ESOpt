def priority(el, n, w):
    frequencies = {i: el.count(i) for i in set(el)}
    total_non_zeros = sum(frequencies[i] for i in set(el) if i!= 0)
    
    if total_non_zeros == 0:
        return 1.0
    
    max_consecutive_non_zeros = 0
    max_consecutive_zeros = 0
    current_consecutive_non_zeros = 0
    current_consecutive_zeros = 0
    last_non_zero_idx = -1
    distinct_non_zeros = 0
    
    for i in range(n):
        if el[i]!= 0:
            current_consecutive_zeros = 0
            current_consecutive_non_zeros += 1
            max_consecutive_non_zeros = max(max_consecutive_non_zeros, current_consecutive_non_zeros)
            if el[i] not in frequencies or frequencies[el[i]] <= 0:
                distinct_non_zeros += 1
            frequencies[el[i]] -= 1
            last_non_zero_idx = i
        else:
            current_consecutive_non_zeros = 0
            current_consecutive_zeros += 1
            max_consecutive_zeros = max(max_consecutive_zeros, current_consecutive_zeros)
    
    if last_non_zero_idx == -1:  # all zeros
        distance_penalty = 1.0
    else:
        distance_penalty = 1.0 - (last_non_zero_idx / n)
    
    reciprocal_non_zeros = sum(1 / (i + 2) if i <= n - 1 else 1 for i in range(max_consecutive_non_zeros))
    reciprocal_zeros = sum(1 / (i + 2) if i <= max_consecutive_zeros else 1 for i in range(max_consecutive_zeros))
    gap_penalty = - (distinct_non_zeros / w + reciprocal_non_zeros / n + reciprocal_zeros / n + distance_penalty)
    
    return gap_penalty
