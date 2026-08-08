def priority(el, n, w):
    count = [0] * (n + 1)
    unique_count = {}
    freq = {}
    for i, e in enumerate(el):
        if e > 0:
            count[i] += 1
            if e not in unique_count:
                unique_count[e] = 1
                freq[1] = freq.get(1, 0) + 1
            else:
                unique_count[e] += 1
                freq[count[i]] = freq.get(count[i], 0) + 1

    score_unique = len(unique_count) * 3
    score_increasing = sum(count[i] < count[i + 1] for i in range(n - 1))
    if freq:
        max_freq = max(freq.values())
        max_repeated = max(count)
        if max_freq > w or max_repeated > max_freq:
            score_unique -= (max_freq - w) * 15
    if len(set(el)) == 1 or w == len(set(el)):
        score_bonus = 0
    else:
        score_bonus = score_increasing / max(count) if max(count) > 0 else score_increasing

    return (score_unique + score_bonus) / (n * (w + 1))
