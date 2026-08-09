# source: /home/zhi/Agentic-ESOpt/cache/active_runs/construct_asp_train_es_sigma0.001_alpha0.0005_construct_asp_train_es_full_reload_sigma1e-3_alpha5e-4_rep3_rerun_agentic_esopt_k1_asp_rep3_8gpu_20260717_101800/results/pops_best/population_generation_25.json
# run_id: construct_asp_train_es_full_reload_sigma1e-3_alpha5e-4_rep3_rerun_agentic_esopt_k1_asp_rep3_8gpu_20260717_101800
# train_objective: -2760.0
# m1m2_multiplier: 1.0

def priority(el, n, w):
    # Calculate the frequency of the first digit in the vector
    first_digit_frequency = 1 - (1 - sum(1 for x in el if x == min(el)!= 0) / len(el) if el else 0)
    
    # Calculate the score based on the number of different combinations of consecutive elements in the vector
    consecutive_combinations_score = 1 - (2 * sum(1 for i in range(len(el)-1) if el[i] == el[i+1] == 2) + 
                                       3 * sum(1 for i in range(len(el)-1) if el[i] == el[i+1] == 1) + 
                                       sum(1 for i in range(len(el)-1) if el[i] == el[i+1] == 0)) / len(el)
    
    # Calculate the ideal state frequency
    ideal_state_frequency = sum(1 for x in el if x == max(set(el))!= 0) / len(el)
    
    # Calculate the score based on the similarity between the vector's frequency and the ideal state
    frequency_similarity_score = 1 - abs(sum(1 for x in el if x == min(el)!= 0) / len(el) - ideal_state_frequency)
    
    # Combine the scores and apply different weights to them
    return first_digit_frequency * 0.4 + consecutive_combinations_score * 0.3 + frequency_similarity_score * 0.3
