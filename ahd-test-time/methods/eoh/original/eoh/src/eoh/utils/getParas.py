
class Paras():
    def __init__(self):
        #####################
        ### General settings  ###
        #####################
        self.method = 'eoh'
        self.problem = 'tsp_construct'
        self.selection = None
        self.management = None
        self.data_split = 'train'
        self.problem_data_root = None

        #####################
        ###  EC settings  ###
        #####################
        self.ec_pop_size = 5  # number of algorithms in each population, default = 10
        self.ec_n_pop = 5 # number of populations, default = 10
        self.ec_operators = None # evolution operators: ['e1','e2','m1','m2'], default = ['e1','m1']
        self.ec_m = 2  # number of parents for 'e1' and 'e2' operators, default = 2
        self.ec_operator_weights = None  # weights for operators, i.e., the probability of use the operator in each iteration, default = [1,1,1,1]
        self.ec_operator_attempts = 1  # number of offspring attempts per operator per generation
        self.ec_m1m2_multiplier = 1.0  # m1/m2 offspring count multiplier relative to population size
        # All four AHD methods share this EoH runtime. Sampling
        # repeatedly invokes i1 without feeding prior candidates into prompts.
        self.ec_run_mode = 'eoh'  # ['eoh', 'sample', 'sample_es']
        self.sample_total = 1000
        self.sample_batch_size = 20
        self.sample_resume = False
        
        #####################
        ### LLM settings  ###
        #####################
        self.llm_use_local = False  # if use local model
        self.llm_local_url = None  # your local server 'http://127.0.0.1:11012/completions'
        self.llm_local_timeout = 180.0
        self.llm_api_endpoint = None # endpoint for remote LLM, e.g., api.deepseek.com
        self.llm_api_key = None  # API key for remote LLM, e.g., sk-xxxx
        self.llm_model = None  # model type for remote LLM, e.g., deepseek-chat
        self.llm_es_enabled = False  # if update the local LLM itself with evolutionary strategy
        self.llm_es_operators = ['e1', 'e2', 'm1', 'm2']  # EoH operators that trigger model ES updates
        self.llm_es_directions = 8  # number of perturbation directions per ES model update
        self.llm_es_sigma = 1e-3  # legacy alias used by migrated launchers
        self.llm_es_sigma_start = 1e-3
        self.llm_es_sigma_end = 1e-3
        self.llm_es_sigma_schedule = 'constant'  # ['constant', 'linear', 'cosine']
        self.llm_es_sigma_warmup_steps = 0
        self.llm_es_sigma_schedule_plateau_fraction = 0.0  # legacy schedule input
        self.llm_es_alpha = 5e-4  # model update step size
        self.llm_es_reward_normalization = 'zscore'
        self.llm_es_reward_normalization_ddof = 0
        self.llm_es_reward_normalization_eps = 1e-8
        self.llm_es_reward_mode = 'improvement'  # ['improvement', 'negative_objective']
        self.llm_es_parameter_scope = 'full'  # ['full', 'all_linear', 'lora']
        self.llm_es_target_modules = None
        self.llm_es_seed = 2024
        self.llm_es_engine_urls = None
        self.llm_es_max_workers = None
        self.llm_es_reward_floor = -1.0
        self.llm_es_invalid_reward_strategy = 'current'  # ['current', 'zero']
        self.llm_es_batch_relative_invalid_reward = False
        self.llm_es_invalid_reward_margin = 1.0
        self.llm_es_invalid_reward_fallback_fraction = 0.01
        self.llm_es_invalid_reward_min_gap = 1.0
        self.llm_es_disable_update = False
        self.llm_es_history_path = None
        self.llm_es_resume_history = None

        #####################
        ###  Exp settings  ###
        #####################
        self.exp_debug_mode = False  # if debug
        self.exp_output_path = "./"  # default folder for ael outputs
        self.exp_use_seed = False
        self.exp_seed_path = "./seeds/seeds.json"
        self.exp_use_continue = False
        self.exp_continue_id = 0
        self.exp_continue_path = "./results/pops/population_generation_0.json"
        self.exp_n_proc = 1
        
        #####################
        ###  Evaluation settings  ###
        #####################
        self.eva_timeout = 30
        self.eva_numba_decorator = False
        self.eva_invalid_objective = float("inf")
        self.evaluation_seed = 1234


    def set_parallel(self):
        import multiprocessing
        num_processes = multiprocessing.cpu_count()
        if self.exp_n_proc == -1 or self.exp_n_proc > num_processes:
            self.exp_n_proc = num_processes
            print(f"Set the number of proc to {num_processes} .")
    
    def set_ec(self):    
        
        if self.method != 'eoh':
            raise ValueError(f"Only EoH is maintained, got method={self.method!r}.")
        if self.management is None:
            self.management = 'pop_greedy'
        
        if self.selection == None:
            self.selection = 'prob_rank'
            
        
        if self.ec_operators is None:
            self.ec_operators = ['e1', 'e2', 'm1', 'm2']
        if self.ec_operator_weights is None:
            self.ec_operator_weights = [1] * len(self.ec_operators)
            
    def set_evaluation(self):
        # Initialize evaluation settings
        if self.problem == 'tsp_construct':
            self.eva_timeout = 20
                
    def set_paras(self, *args, **kwargs):
        
        # Map paras
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
              
        # Identify and set parallel 
        self.set_parallel()
        
        # Initialize method and ec settings
        self.set_ec()
        
        # Initialize evaluation settings
        self.set_evaluation()




if __name__ == "__main__":

    # Create an instance of the Paras class
    paras_instance = Paras()

    # Setting parameters using the set_paras method
    paras_instance.set_paras(llm_use_local=True, llm_local_url='http://example.com', ec_pop_size=8)

    # Accessing the updated parameters
    print(paras_instance.llm_use_local)  # Output: True
    print(paras_instance.llm_local_url)  # Output: http://example.com
    print(paras_instance.ec_pop_size)    # Output: 8
            
            
            
