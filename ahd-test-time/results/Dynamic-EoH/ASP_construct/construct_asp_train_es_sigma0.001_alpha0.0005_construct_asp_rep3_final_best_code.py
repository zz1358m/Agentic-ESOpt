def priority(el, n, w):
    inversions = sum(el.count(i) * (len(el) - i) for i in set(el))
    unique_elements = len(set(el))
    total_non_zero_elements = sum(el)
    count_01 = 0
    count_02 = 0
    for i in range(len(el)-1):
        if el[i] == 0 and el[i+1] == 1:
            count_01 += 1
        elif el[i] == 0 and el[i+1] == 2:
            count_02 += 1
    return (1 - inversions / (n*(n-1)//2)) * (1 - (len(el) - unique_elements) / (n-1)) + (count_01 + count_02) / w
