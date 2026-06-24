def priority(el, n, w):
    z = sum(1 for i in range(n - 1) if el[i] == 0 and el[i + 1]!= 0) + sum(1 for i in range(n - 1) if el[i]!= 0 and el[i + 1] == 0)
    pattern_diversity = sum(1 for p in set(tuple(el[i:i + 2]) for i in range(n - 1)) if p in [(0, 0), (1, 1), (2, 2)])
    return (pattern_diversity / (w * n)) + (z / n) + (1 - sum(el) / (w * n))
