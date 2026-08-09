# Archived from /home/zhi/Agentic-ESOpt/cache/active_runs/aco_tsp_train_es_sigma0.001_alpha0.0005_aco_tsp_train_es_full_reload_sigma1e-3_alpha5e-4_rep2_k1_decay_full_all6_3rep_reload8_20260706_153046/results/pops_best/population_generation_25.json
# run_id: aco_tsp_train_es_full_reload_sigma1e-3_alpha5e-4_rep2_k1_decay_full_all6_3rep_reload8_20260706_153046
# train_objective: 5.81302
# m1m2_multiplier: 1.0
# sigma_schedule: cosine
# final_model_es_sigma: None

import numpy as np
import networkx as nx

def heuristics_v2(distance_matrix):
    num_trees = 2000
    num_nodes = distance_matrix.shape[0]
    score_count = np.zeros((num_nodes, num_nodes))
    seen = np.zeros((num_nodes, num_nodes))

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            G.add_edge(i, j, weight=distance_matrix[i, j])

    for _ in range(num_trees):
        tree = np.random.permutation(list(range(num_nodes)))
        total_dist = 0
        for i in range(num_nodes):
            min_dist = np.inf
            index = -1
            for j in range(i+1, num_nodes):
                if distance_matrix[tree[i], tree[j]] > 0:
                    total_dist += distance_matrix[tree[i], tree[j]]
                    if distance_matrix[tree[i], tree[j]] < min_dist:
                        min_dist = distance_matrix[tree[i], tree[j]]
                        index = j
            if index!=-1:
                score_count[tree[i], tree[index]] += 1
                score_count[tree[index], tree[i]] += 1
                seen[tree[i], tree[index]] += 1
                seen[tree[index], tree[i]] += 1

    # Consider using distances as a proxy for similarity between nodes to encourage locally-structured paths
    dist_matrix = distance_matrix.copy()
    dist_matrix[dist_matrix == 0] = np.inf
    max_dist = np.max(dist_matrix, axis=1, keepdims=True)
    max_dist = np.repeat(max_dist, num_nodes, axis=1)
    dist_matrix = dist_matrix / max_dist
    score_count = score_count / dist_matrix

    np.fill_diagonal(score_count, 0)

    max_score = np.max(score_count, axis=1, keepdims=True)
    max_score = np.repeat(max_score, num_nodes, axis=1)
    score_count = score_count / max_score

    # Perform exponential decay and popularity bias as in the original algorithms
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if distance_matrix[i, j] == 0:
                score_count[i, j] = 0
            else:
                leverage = score_count[i, j]
                bias = np.log(seen[i, j] + 1) / np.log(num_trees)
                exponent = 1.8
                base = 2
                score_count[i, j] = leverage * bias * np.exp(-(distance_matrix[i, j] ** 2) / base)

    return score_count
