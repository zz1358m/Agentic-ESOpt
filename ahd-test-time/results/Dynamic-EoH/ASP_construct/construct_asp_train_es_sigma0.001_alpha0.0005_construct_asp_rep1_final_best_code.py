def priority(el, n, w):
    def get_average_deviation(el):
        non_zero_elements = [el[i] for i in range(n) if el[i]!= 0]
        if not non_zero_elements:
            return 0
        non_zero_elements = [val / 3 for val in non_zero_elements]  # normalize to {0, 1/3, 2/3}
        average_deviation = sum(abs(non_zero_elements[i] - (non_zero_elements[i-1] if i > 0 else 0)) for i in range(len(non_zero_elements))) / len(non_zero_elements)
        return average_deviation

    zero_to_nonzero_ratio = sum(1 for i in range(n) if (el[i-1] == 0 and el[i]!= 0) or (el[i-1]!= 0 and el[i] == 0)) / w if w > 0 else 0
    decay_factor = 1.5 ** (-zero_to_nonzero_ratio / (w * len(el)))
    return (0.7 * get_average_deviation(el)) + (0.3 * (zero_to_nonzero_ratio * decay_factor))
