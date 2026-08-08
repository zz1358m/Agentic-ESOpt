def priority(el, n, w):
    def count_subsets(el):
        subsets = []
        for i in range(0, n, n//3):
            subsets.append(tuple([x for j, x in enumerate(el) if i <= j < i + n//3]))
        return subsets

    def subset_frequency(subset):
        return max(subset.count(0), subset.count(1), subset.count(2)) 

    def transition_score(el):
        transitions = 0
        for i in range(n-1):
            if el[i]!= el[i+1]:
                transitions += 1
        return transitions

    frequency = 0
    for subset in count_subsets(el):
        if subset_frequency(subset) > 0:
            frequency += subset_frequency(subset) / len(subset)
    
    locality = 0.3 * (len(el) / sum(subset_frequency(subset) for subset in count_subsets(el))) + 0.7 * transition_score(el)

    def calculate_variance(freqs):
        mean = sum(freqs) / len(freqs)
        squared_diffs = [(f - mean) ** 2 for f in freqs]
        return sum(squared_diffs) / len(squared_diffs)

    freqs = [subset_frequency(subset) for subset in count_subsets(el) if subset_frequency(subset) > 0]
    var = calculate_variance(freqs)
    return locality - (freqs[0] / sum(freqs) if freqs else 0) * var * 0.1
