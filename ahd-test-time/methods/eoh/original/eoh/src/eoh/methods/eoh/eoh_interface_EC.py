import numpy as np
import time
from .eoh_evolution import Evolution
import warnings
from joblib import Parallel, delayed
from .evaluator_accelerate import add_numba_decorator
import re
import concurrent.futures
import multiprocessing as mp
import os
import random
from pathlib import Path

from es import ModelESClient
from es.run_state import (
    atomic_write_history,
    map_endpoint_serial,
    normalize_sigma_schedule,
    read_history,
    sigma_at_step,
)

DEFAULT_INVALID_OBJECTIVE = float("inf")

class InterfaceEC():
    def __init__(self, pop_size, m, api_endpoint, api_key, llm_model,llm_use_local,llm_local_url, debug_mode, interface_prob, select,n_p,timeout,use_numba,**kwargs):

        # LLM settings
        self.pop_size = pop_size
        self.interface_eval = interface_prob
        prompts = interface_prob.prompts
        self.evol = Evolution(api_endpoint, api_key, llm_model,llm_use_local,llm_local_url, debug_mode,prompts, **kwargs)
        self.local_evolutions = []
        self.m = m
        self.debug = debug_mode

        if not self.debug:
            warnings.filterwarnings("ignore")

        self.select = select
        self.n_p = n_p
        
        self.timeout = timeout
        self.use_numba = use_numba
        self.paras = kwargs.get("paras", None)
        self.generation_timeout = max(
            float(getattr(self.paras, "llm_local_timeout", 180.0)),
            float(self.timeout) + 15.0,
        )
        self.invalid_objective = float(getattr(self.paras, "eva_invalid_objective", DEFAULT_INVALID_OBJECTIVE))
        self.llm_use_local = llm_use_local
        self.llm_local_url = llm_local_url
        self.model_es_enabled = bool(getattr(self.paras, "llm_es_enabled", False))
        self.model_es_client = None
        self.model_es_clients = []
        self.model_es_evolutions = []
        self.model_es_rng = random.Random(int(getattr(self.paras, "llm_es_seed", 2024)))
        self.current_generation_index = 0
        self.total_generations = int(getattr(self.paras, "ec_n_pop", 1))
        configured_history_path = getattr(self.paras, "llm_es_history_path", None)
        if configured_history_path:
            self.model_es_history_path = Path(configured_history_path).expanduser().resolve()
        else:
            output_root = Path(getattr(self.paras, "exp_output_path", "./")).expanduser().resolve()
            self.model_es_history_path = output_root / "results" / "es" / "history.json"
        self.model_es_history = []

        if self.llm_use_local and not self.model_es_enabled:
            local_urls = [url.strip() for url in str(self.llm_local_url).split(",") if url.strip()]
            if len(local_urls) > 1:
                self.local_evolutions = [
                    Evolution(api_endpoint, api_key, llm_model, llm_use_local, url, debug_mode, prompts, **kwargs)
                    for url in local_urls
                ]

        if self.model_es_enabled:
            if not self.llm_use_local:
                raise ValueError("llm_es_enabled=True requires llm_use_local=True.")
            es_urls = getattr(self.paras, "llm_es_engine_urls", None) or [self.llm_local_url]
            self.model_es_clients = [
                ModelESClient(url, timeout=max(float(self.timeout) * 20, 600.0))
                for url in es_urls
            ]
            self.model_es_evolutions = [
                Evolution(api_endpoint, api_key, llm_model, llm_use_local, url, debug_mode, prompts, **kwargs)
                for url in es_urls
            ]
            if len(es_urls) > 1:
                self.local_evolutions = self.model_es_evolutions
            init_info = []
            for client in self.model_es_clients:
                init_info.append(client.init(
                    parameter_scope=getattr(self.paras, "llm_es_parameter_scope", "full"),
                    target_modules=getattr(self.paras, "llm_es_target_modules", None),
                    verbose=not self.debug,
                ))
            self.model_es_client = self.model_es_clients[0]
            print(f"- Model ES initialized: {init_info}")
            self._restore_model_es_history()
        
    def code2file(self,code):
        with open("./ael_alg.py", "w") as file:
        # Write the code to the file
            file.write(code)
        return 
    
    def add2pop(self,population,offspring):
        for ind in population:
            if ind['objective'] == offspring['objective']:
                if self.debug:
                    print("duplicated result, retrying ... ")
                return False
        population.append(offspring)
        return True
    
    def check_duplicate(self,population,code):
        for ind in population:
            if code == ind['code']:
                return True
        return False

    def check_duplicate_objective(self, population, objective):
        for ind in population:
            if objective == ind.get('objective'):
                return True
        return False

    # def population_management(self,pop):
    #     # Delete the worst individual
    #     pop_new = heapq.nsmallest(self.pop_size, pop, key=lambda x: x['objective'])
    #     return pop_new
    
    # def parent_selection(self,pop,m):
    #     ranks = [i for i in range(len(pop))]
    #     probs = [1 / (rank + 1 + len(pop)) for rank in ranks]
    #     parents = random.choices(pop, weights=probs, k=m)
    #     return parents

    def population_generation(self):
        population = []
        max_attempts = max(1, int(getattr(self.paras, "ec_init_attempts", 1)))
        attempts = 0

        while len(population) < self.pop_size and (attempts < max_attempts or len(population) == 0):
            attempts += 1
            _, pop = self.get_algorithm([], 'i1')
            valid_count = 0
            added_count = 0
            for p in pop:
                if len(population) >= self.pop_size:
                    break
                if p.get('objective') is None or not np.isfinite(float(p.get('objective', self.invalid_objective))):
                    p['objective'] = self.invalid_objective
                    continue
                valid_count += 1
                if self.check_duplicate_objective(population, p.get('objective')):
                    continue
                population.append(p)
                added_count += 1
            print(
                f"Init attempt {attempts}: valid={valid_count}/{len(pop)}, "
                f"added={added_count}, population={len(population)}/{self.pop_size}",
                flush=True,
            )

        if len(population) < self.pop_size:
            print(f"Warning: initialized {len(population)} of {self.pop_size} individuals after {attempts} attempts.")
             
        return population
    
    def population_generation_seed(self,seeds,n_p):

        population = []

        fitness = Parallel(n_jobs=n_p)(delayed(self.interface_eval.evaluate)(seed['code']) for seed in seeds)

        for i in range(len(seeds)):
            try:
                seed_alg = {
                    'algorithm': seeds[i]['algorithm'],
                    'code': seeds[i]['code'],
                    'objective': None,
                    'other_inf': None
                }

                obj = np.array(fitness[i])
                seed_alg['objective'] = np.round(obj, 5)
                population.append(seed_alg)

            except Exception as e:
                print("Error in seed algorithm")
                exit()

        print("Initiliazation finished! Get "+str(len(seeds))+" seed algorithms")

        return population
    

    def _get_alg(self,pop,operator,evol=None):
        evol = self.evol if evol is None else evol
        offspring = {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': None
        }
        if operator == "i1":
            parents = None
            [offspring['code'],offspring['algorithm']] =  evol.i1()
        elif operator == "e1":
            parents = self.select.parent_selection(pop,self.m)
            [offspring['code'],offspring['algorithm']] = evol.e1(parents)
        elif operator == "e2":
            parents = self.select.parent_selection(pop,self.m)
            [offspring['code'],offspring['algorithm']] = evol.e2(parents)
        elif operator == "m1":
            parents = self.select.parent_selection(pop,1)
            [offspring['code'],offspring['algorithm']] = evol.m1(parents[0])
        elif operator == "m2":
            parents = self.select.parent_selection(pop,1)
            [offspring['code'],offspring['algorithm']] = evol.m2(parents[0])
        else:
            print(f"Evolution operator [{operator}] has not been implemented ! \n") 

        return parents, offspring

    def get_offspring(self, pop, operator, evol=None):

        try:
            p, offspring = self._get_alg(pop, operator, evol=evol)
            
            if self.use_numba:
                
                # Regular expression pattern to match function definitions
                pattern = r"def\s+(\w+)\s*\(.*\):"

                # Search for function definitions in the code
                match = re.search(pattern, offspring['code'])

                function_name = match.group(1)

                code = add_numba_decorator(program=offspring['code'], function_name=function_name)
            else:
                code = offspring['code']

            n_retry= 1
            while self.check_duplicate(pop, offspring['code']):
                
                n_retry += 1
                if self.debug:
                    print("duplicated code, wait 1 second and retrying ... ")
                    
                p, offspring = self._get_alg(pop, operator, evol=evol)

                if self.use_numba:
                    # Regular expression pattern to match function definitions
                    pattern = r"def\s+(\w+)\s*\(.*\):"

                    # Search for function definitions in the code
                    match = re.search(pattern, offspring['code'])

                    function_name = match.group(1)

                    code = add_numba_decorator(program=offspring['code'], function_name=function_name)
                else:
                    code = offspring['code']
                    
                if n_retry > 1:
                    break
                
                
            fitness = self._evaluate_code_with_timeout(code)
            offspring['objective'] = np.round(fitness, 5)
                

        except Exception as e:

            offspring = {
                'algorithm': None,
                'code': None,
                'objective': self.invalid_objective,
                'other_inf': {'error': str(e)}
            }
            p = None

        # Round the objective values
        return p, offspring

    def _get_offspring_from_parent(self, parent, operator, evol=None):
        evol = self.evol if evol is None else evol
        offspring = {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': None
        }
        if operator == "m1":
            offspring['code'], offspring['algorithm'] = evol.m1(parent)
        elif operator == "m2":
            offspring['code'], offspring['algorithm'] = evol.m2(parent)
        else:
            raise ValueError(f"Model ES only supports mutation operators, got {operator}")

        code = self._prepare_code_for_evaluation(offspring['code'])
        fitness = self._evaluate_code_with_timeout(code)
        offspring['objective'] = np.round(fitness, 5)
        return offspring

    def _get_offspring_from_parents(self, parents, operator, evol=None):
        evol = self.evol if evol is None else evol
        offspring = {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': None
        }
        if operator == "e1":
            offspring['code'], offspring['algorithm'] = evol.e1(parents)
        elif operator == "e2":
            offspring['code'], offspring['algorithm'] = evol.e2(parents)
        elif operator in {"m1", "m2"}:
            return self._get_offspring_from_parent(parents[0], operator, evol=evol)
        else:
            raise ValueError(f"Unsupported operator for Model ES: {operator}")

        code = self._prepare_code_for_evaluation(offspring['code'])
        fitness = self._evaluate_code_with_timeout(code)
        offspring['objective'] = np.round(fitness, 5)
        return offspring

    def _parents_reference_objective(self, parents):
        if not parents:
            return None
        parent_objectives = []
        for parent in parents:
            objective = parent.get('objective')
            if objective is None:
                continue
            objective = float(objective)
            if np.isfinite(objective):
                parent_objectives.append(objective)
        if not parent_objectives:
            return None
        return min(parent_objectives)

    def _evaluate_code_with_timeout(self, code):
        ctx = mp.get_context("fork")
        queue = ctx.Queue(maxsize=1)

        def run_eval(out_queue):
            try:
                with open(os.devnull, "w") as devnull:
                    old_stdout = os.dup(1)
                    old_stderr = os.dup(2)
                    try:
                        os.dup2(devnull.fileno(), 1)
                        os.dup2(devnull.fileno(), 2)
                        out_queue.put(self.interface_eval.evaluate(code))
                    finally:
                        os.dup2(old_stdout, 1)
                        os.dup2(old_stderr, 2)
                        os.close(old_stdout)
                        os.close(old_stderr)
            except Exception:
                out_queue.put(None)

        proc = ctx.Process(target=run_eval, args=(queue,))
        proc.daemon = True
        proc.start()
        proc.join(self.timeout)
        if proc.is_alive():
            proc.terminate()
            proc.join(2)
            if proc.is_alive():
                proc.kill()
                proc.join()
            return self.invalid_objective

        try:
            fitness = queue.get_nowait()
        except Exception:
            return self.invalid_objective
        if fitness is None or not np.isfinite(float(fitness)):
            return self.invalid_objective
        return fitness

    def _evaluate_offspring_batch_with_timeout(self, pairs):
        if not pairs:
            return []

        ctx = mp.get_context("fork")
        queue = ctx.Queue()
        results = [self.invalid_objective] * len(pairs)
        running = {}
        next_idx = 0
        max_workers = max(1, int(self.n_p))

        def run_eval(idx, code, out_queue):
            try:
                with open(os.devnull, "w") as devnull:
                    old_stdout = os.dup(1)
                    old_stderr = os.dup(2)
                    try:
                        os.dup2(devnull.fileno(), 1)
                        os.dup2(devnull.fileno(), 2)
                        out_queue.put((idx, self.interface_eval.evaluate(code)))
                    finally:
                        os.dup2(old_stdout, 1)
                        os.dup2(old_stderr, 2)
                        os.close(old_stdout)
                        os.close(old_stderr)
            except Exception:
                out_queue.put((idx, None))

        def launch_more():
            nonlocal next_idx
            while next_idx < len(pairs) and len(running) < max_workers:
                _, offspring = pairs[next_idx]
                try:
                    code = self._prepare_code_for_evaluation(offspring['code'])
                except Exception:
                    results[next_idx] = self.invalid_objective
                    next_idx += 1
                    continue
                proc = ctx.Process(target=run_eval, args=(next_idx, code, queue))
                proc.daemon = True
                proc.start()
                running[next_idx] = (proc, time.time())
                next_idx += 1

        launch_more()
        while running:
            while True:
                try:
                    idx, fitness = queue.get_nowait()
                except Exception:
                    break
                if fitness is not None and np.isfinite(float(fitness)):
                    results[idx] = fitness

            for idx, (proc, started_at) in list(running.items()):
                if not proc.is_alive():
                    proc.join()
                    running.pop(idx, None)
                    launch_more()
                    continue
                if time.time() - started_at >= self.timeout:
                    proc.terminate()
                    proc.join(2)
                    if proc.is_alive():
                        proc.kill()
                        proc.join()
                    results[idx] = self.invalid_objective
                    running.pop(idx, None)
                    launch_more()
            time.sleep(0.1)

        return results

    def _prepare_code_for_evaluation(self, code):
        if not self.use_numba:
            return code

        pattern = r"def\s+(\w+)\s*\(.*\):"
        match = re.search(pattern, code)
        if match is None:
            return code
        function_name = match.group(1)
        return add_numba_decorator(program=code, function_name=function_name)

    def _population_worst_objective(self, pop):
        objectives = []
        for individual in pop:
            objective = individual.get('objective')
            if objective is None:
                continue
            objective = float(objective)
            if np.isfinite(objective):
                objectives.append(objective)
        if not objectives:
            return None
        return max(objectives)

    def _effective_reward_objective(
        self,
        objective,
        population_worst_objective,
        parent_objective=None,
        reward_floor=None,
    ):
        objective_is_valid = objective is not None
        if objective_is_valid:
            objective = float(objective)
            objective_is_valid = np.isfinite(objective)
        if population_worst_objective is None:
            return objective, objective_is_valid
        if not objective_is_valid or objective > population_worst_objective:
            effective_objective = float(population_worst_objective)
            if parent_objective is not None and reward_floor is not None:
                parent_objective = float(parent_objective)
                if np.isfinite(parent_objective):
                    # Keep the worst fallback no worse than the configured reward floor
                    # relative to the current parent. With the default floor -1, this
                    # is equivalent to max(parent - population_worst, -1) in reward space.
                    effective_objective = min(effective_objective, parent_objective - float(reward_floor))
            return effective_objective, True
        return objective, True

    def _objective_to_reward(self, offspring, parent, population_worst_objective=None):
        objective = offspring.get('objective')
        reward_floor = float(getattr(self.paras, "llm_es_reward_floor", -1.0))
        reward_mode = str(getattr(self.paras, "llm_es_reward_mode", "improvement")).lower()
        if reward_mode == "negative_objective":
            objective, objective_is_valid = self._effective_reward_objective(objective, population_worst_objective)
            reward = -objective if objective_is_valid else reward_floor
            return reward if population_worst_objective is not None else max(reward, reward_floor)

        parent_objective = None if parent is None else parent.get('objective')
        if parent_objective is None:
            objective, objective_is_valid = self._effective_reward_objective(objective, population_worst_objective)
            reward = -objective if objective_is_valid else reward_floor
            return reward if population_worst_objective is not None else max(reward, reward_floor)
        parent_objective = float(parent_objective)
        if not np.isfinite(parent_objective):
            objective, objective_is_valid = self._effective_reward_objective(objective, population_worst_objective)
            reward = -objective if objective_is_valid else reward_floor
            return reward if population_worst_objective is not None else max(reward, reward_floor)
        objective, objective_is_valid = self._effective_reward_objective(
            objective,
            population_worst_objective,
            parent_objective=parent_objective,
            reward_floor=reward_floor,
        )
        reward = parent_objective - objective if objective_is_valid else reward_floor
        return reward if population_worst_objective is not None else max(reward, reward_floor)

    def _sample_model_es_seeds(self, n=None):
        default_directions = int(getattr(self.paras, "llm_es_directions", self.pop_size))
        directions = default_directions if n is None else int(n)
        directions = max(1, directions)
        return [self.model_es_rng.randrange(0, 2**31 - 1) for _ in range(directions)]

    def _operator_uses_model_es(self, operator):
        if not self.model_es_enabled:
            return False
        operators = getattr(self.paras, "llm_es_operators", ['m1', 'm2'])
        return operator in operators

    def _operator_offspring_count(self, operator, base_count=None):
        base_count = self.pop_size if base_count is None else int(base_count)
        if operator in {"m1", "m2"}:
            multiplier = float(getattr(self.paras, "ec_m1m2_multiplier", 1.0))
            return max(1, int(round(base_count * multiplier)))
        return max(1, base_count)

    def set_generation_context(self, generation_index, total_generations=None):
        self.current_generation_index = max(0, int(generation_index))
        if total_generations is not None:
            self.total_generations = max(1, int(total_generations))

    def _current_model_es_sigma(self):
        sigma_start = float(getattr(self.paras, "llm_es_sigma_start", 1e-3))
        sigma_end = float(getattr(self.paras, "llm_es_sigma_end", sigma_start))
        schedule = normalize_sigma_schedule(getattr(self.paras, "llm_es_sigma_schedule", "constant"))
        warmup_steps = int(getattr(self.paras, "llm_es_sigma_warmup_steps", 0))
        return sigma_at_step(
            sigma_start=sigma_start,
            sigma_end=sigma_end,
            step=self.current_generation_index,
            total_steps=self.total_generations,
            schedule=schedule,
            warmup_steps=int(warmup_steps),
        )

    def _restore_model_es_history(self):
        source = getattr(self.paras, "llm_es_resume_history", None)
        if not source:
            print(f"- Model ES history: {self.model_es_history_path}")
            return

        source_path = Path(source).expanduser().resolve()
        source_history = read_history(source_path)
        records = [
            row
            for row in source_history
            if isinstance(row.get("seeds"), list)
            and isinstance(row.get("rewards"), list)
            and len(row["seeds"]) == len(row["rewards"])
            and len(row["seeds"]) > 0
        ]
        expected_rng = random.Random(int(getattr(self.paras, "llm_es_seed", 2024)))
        replayed_updates = 0
        for record_index, record in enumerate(records):
            seeds = [int(seed) for seed in record["seeds"]]
            expected = [expected_rng.randrange(0, 2**31 - 1) for _ in seeds]
            if seeds != expected:
                raise ValueError(
                    f"AHD ES history seed mismatch at update {record_index}; "
                    "resume with the original --seed/operator order/k."
                )
            # A disabled update still consumed these seeds in the original run.
            # Advance/validate the stream, but do not mutate model weights.
            if record.get("update_applied", True) is False:
                continue
            kwargs = {
                "seeds": seeds,
                "rewards": [float(reward) for reward in record["rewards"]],
                "alpha": float(record.get("alpha", getattr(self.paras, "llm_es_alpha", 5e-4))),
                "reward_normalization": str(
                    record.get(
                        "reward_normalization",
                        getattr(self.paras, "llm_es_reward_normalization", "zscore"),
                    )
                ),
                "reward_normalization_ddof": int(
                    record.get(
                        "reward_normalization_ddof",
                        getattr(self.paras, "llm_es_reward_normalization_ddof", 0),
                    )
                ),
                "reward_normalization_eps": float(
                    record.get(
                        "reward_normalization_eps",
                        getattr(self.paras, "llm_es_reward_normalization_eps", 1e-8),
                    )
                ),
            }
            for client in self.model_es_clients:
                client.update(**kwargs)
            replayed_updates += 1
        self.model_es_rng = expected_rng
        self.model_es_history = list(source_history)
        if source_path != self.model_es_history_path:
            self.model_es_history.append(
                {
                    "resume": {
                        "source": str(source_path),
                        "replayed_samples": len(records),
                        "replayed_updates": replayed_updates,
                    }
                }
            )
        atomic_write_history(self.model_es_history_path, self.model_es_history)
        print(
            f"- Model ES replayed {replayed_updates} updates from {len(records)} "
            f"sample batches at {source_path}; "
            f"history: {self.model_es_history_path}"
        )

    def _get_algorithm_with_model_es(self, pop, operator):
        if len(pop) == 0:
            return [], []

        base_sigma = float(getattr(self.paras, "llm_es_sigma_start", 1e-3))
        end_sigma = float(getattr(self.paras, "llm_es_sigma_end", base_sigma))
        sigma = self._current_model_es_sigma()
        sigma_schedule = normalize_sigma_schedule(
            getattr(self.paras, "llm_es_sigma_schedule", "constant")
        )
        sigma_warmup_steps = getattr(self.paras, "llm_es_sigma_warmup_steps", None)
        alpha = float(getattr(self.paras, "llm_es_alpha", 5e-4))
        base_directions = int(getattr(self.paras, "llm_es_directions", self.pop_size))
        n_directions = self._operator_offspring_count(operator, base_directions)
        seeds = self._sample_model_es_seeds(n_directions)
        rewards = [None] * len(seeds)
        offsprings = [None] * len(seeds)
        out_parents = [None] * len(seeds)
        tasks = []
        population_worst_objective = self._population_worst_objective(pop)

        for i, seed in enumerate(seeds):
            n_parents = self.m if operator in {"e1", "e2"} else 1
            parents = self.select.parent_selection(pop, n_parents)
            tasks.append((i, seed, parents))

        def eval_direction(i, seed, parents):
            engine_id = i % len(self.model_es_clients)
            client = self.model_es_clients[engine_id]
            evol = self.model_es_evolutions[engine_id]
            parent_objective = self._parents_reference_objective(parents)
            reward_parent = None if parent_objective is None else {'objective': parent_objective}
            try:
                client.apply_perturbation(seed=seed, sigma=sigma)
                try:
                    offspring = self._get_offspring_from_parents(parents, operator, evol=evol)
                finally:
                    client.revert_perturbation(seed=seed, sigma=sigma)

                reward = self._objective_to_reward(
                    offspring,
                    reward_parent,
                    population_worst_objective=population_worst_objective,
                )
                offspring['other_inf'] = {
                    'model_es_seed': int(seed),
                    'model_es_sigma': sigma,
                    'model_es_base_sigma': base_sigma,
                    'model_es_end_sigma': end_sigma,
                    'model_es_sigma_schedule': sigma_schedule,
                    'model_es_sigma_warmup_steps': sigma_warmup_steps,
                    'model_es_generation_index': int(getattr(self, "current_generation_index", 0)),
                    'model_es_total_generations': int(getattr(self, "total_generations", 1)),
                    'model_es_reward': float(reward),
                    'model_es_parent_objective': parent_objective,
                    'model_es_population_worst_objective': population_worst_objective,
                    'model_es_operator': operator,
                    'model_es_batch_size': len(seeds),
                    'model_es_base_batch_size': base_directions,
                    'm1m2_multiplier': float(getattr(self.paras, "ec_m1m2_multiplier", 1.0)),
                    'model_es_engine_id': engine_id,
                }
            except Exception as e:
                if self.debug:
                    print(f"Model ES direction failed: seed={seed}, error={e}")
                offspring = {
                    'algorithm': None,
                    'code': None,
                    'objective': self.invalid_objective,
                    'other_inf': {
                        'model_es_seed': int(seed),
                        'model_es_sigma': sigma,
                        'model_es_base_sigma': base_sigma,
                        'model_es_end_sigma': end_sigma,
                        'model_es_sigma_schedule': sigma_schedule,
                        'model_es_sigma_warmup_steps': sigma_warmup_steps,
                        'model_es_generation_index': int(getattr(self, "current_generation_index", 0)),
                        'model_es_total_generations': int(getattr(self, "total_generations", 1)),
                        'model_es_reward': None,
                        'model_es_error': str(e),
                        'model_es_operator': operator,
                        'model_es_batch_size': len(seeds),
                        'model_es_base_batch_size': base_directions,
                        'm1m2_multiplier': float(getattr(self.paras, "ec_m1m2_multiplier", 1.0)),
                        'model_es_engine_id': engine_id,
                    }
                }
                reward = self._objective_to_reward(
                    offspring,
                    reward_parent,
                    population_worst_objective=population_worst_objective,
                )
                offspring['other_inf']['model_es_reward'] = float(reward)
                offspring['other_inf']['model_es_population_worst_objective'] = population_worst_objective
            return i, parents, offspring, float(reward)

        def endpoint_worker(i, _endpoint_index):
            _, seed, parents = tasks[i]
            _, selected_parents, offspring, reward = eval_direction(i, seed, parents)
            return i, (selected_parents, offspring, reward)

        direction_results = map_endpoint_serial(
            endpoints=[str(index) for index in range(len(self.model_es_clients))],
            count=len(tasks),
            worker=endpoint_worker,
        )
        for i, (parents, offspring, reward) in enumerate(direction_results):
            rewards[i] = reward
            offsprings[i] = offspring
            out_parents[i] = parents

        update_info = [
            {
                "ok": True,
                "skipped": True,
                "reason": "llm_es_disable_update",
                "num_seeds": len(seeds),
                "base_num_seeds": base_directions,
                "m1m2_multiplier": float(getattr(self.paras, "ec_m1m2_multiplier", 1.0)),
                "alpha": alpha,
                "sigma": sigma,
                "base_sigma": base_sigma,
                "sigma_schedule": sigma_schedule,
            }
        ] if bool(getattr(self.paras, "llm_es_disable_update", False)) else [
            client.update(
                    seeds=seeds,
                    rewards=rewards,
                    alpha=alpha,
                    reward_normalization=getattr(self.paras, "llm_es_reward_normalization", "zscore"),
                    reward_normalization_ddof=getattr(self.paras, "llm_es_reward_normalization_ddof", 0),
                    reward_normalization_eps=getattr(self.paras, "llm_es_reward_normalization_eps", 1e-8),
                )
                for client in self.model_es_clients
            ]
        if self.debug:
            print(f"Model ES update: {update_info}")

        update_applied = not bool(getattr(self.paras, "llm_es_disable_update", False))
        history_record = {
            "update_index": sum(
                1
                for row in self.model_es_history
                if isinstance(row.get("seeds"), list)
                and isinstance(row.get("rewards"), list)
                and len(row["seeds"]) == len(row["rewards"])
                and len(row["seeds"]) > 0
            ),
            "generation": int(getattr(self, "current_generation_index", 0)),
            "operator": operator,
            "seeds": [int(seed) for seed in seeds],
            "rewards": [float(reward) for reward in rewards],
            "alpha": alpha,
            "sigma": sigma,
            "sigma_start": base_sigma,
            "sigma_end": end_sigma,
            "sigma_schedule": sigma_schedule,
            "sigma_warmup_steps": sigma_warmup_steps,
            "reward_normalization": getattr(self.paras, "llm_es_reward_normalization", "zscore"),
            "reward_normalization_ddof": int(
                getattr(self.paras, "llm_es_reward_normalization_ddof", 0)
            ),
            "reward_normalization_eps": float(
                getattr(self.paras, "llm_es_reward_normalization_eps", 1e-8)
            ),
            "update_applied": update_applied,
            "update": update_info,
        }
        self.model_es_history.append(history_record)
        atomic_write_history(self.model_es_history_path, self.model_es_history)

        return out_parents, offsprings
    # def process_task(self,pop, operator):
    #     result =  None, {
    #             'algorithm': None,
    #             'code': None,
    #             'objective': None,
    #             'other_inf': None
    #         }
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         future = executor.submit(self.get_offspring, pop, operator)
    #         try:
    #             result = future.result(timeout=self.timeout)
    #             future.cancel()
    #             #print(result)
    #         except:
    #             future.cancel()
                
    #     return result

    
    def get_algorithm(self, pop, operator):
        if self._operator_uses_model_es(operator):
            return self._get_algorithm_with_model_es(pop, operator)

        offspring_count = self._operator_offspring_count(operator)
        if self.local_evolutions:
            pairs = [None] * offspring_count
            results = []
            max_workers = min(len(self.local_evolutions), offspring_count)
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            self._get_alg,
                            pop,
                            operator,
                            self.local_evolutions[i % len(self.local_evolutions)],
                        ): i
                        for i in range(offspring_count)
                    }
                    done, not_done = concurrent.futures.wait(futures, timeout=self.generation_timeout)
                    for future in done:
                        i = futures[future]
                        try:
                            pairs[i] = future.result()
                        except Exception:
                            pairs[i] = (None, {
                                'algorithm': None,
                                'code': None,
                                'objective': self.invalid_objective,
                                'other_inf': {'error': 'generation failed'}
                            })
                    for future in not_done:
                        future.cancel()
            except Exception as e:
                if self.debug:
                    print(f"Error: {e}")
                print("Parallel time out .")
            for i, pair in enumerate(pairs):
                if pair is None:
                    pair = (None, {
                        'algorithm': None,
                        'code': None,
                        'objective': self.invalid_objective,
                        'other_inf': {
                            'error': 'generation timeout',
                            'operator_batch_size': offspring_count,
                            'm1m2_multiplier': float(getattr(self.paras, "ec_m1m2_multiplier", 1.0)),
                        }
                    })
                    pairs[i] = pair
            to_eval = [
                pair for pair in pairs
                if pair[1].get('code') is not None
            ]
            fitness = self._evaluate_offspring_batch_with_timeout(to_eval)
            fitness_iter = iter(fitness)
            for p, off in pairs:
                if off.get('code') is not None:
                    off['objective'] = np.round(next(fitness_iter), 5)
                results.append((p, off))
        else:
            results = []
            try:
                results = Parallel(n_jobs=self.n_p,timeout=self.timeout+15)(
                    delayed(self.get_offspring)(pop, operator) for _ in range(offspring_count)
                )
            except Exception as e:
                if self.debug:
                    print(f"Error: {e}")
                print("Parallel time out .")

        time.sleep(2)


        out_p = []
        out_off = []

        for p, off in results:
            out_p.append(p)
            out_off.append(off)
            if self.debug:
                print(f">>> check offsprings: \n {off}")
        return out_p, out_off
    # def get_algorithm(self,pop,operator, pop_size, n_p):
        
    #     # perform it pop_size times with n_p processes in parallel
    #     p,offspring = self._get_alg(pop,operator)
    #     while self.check_duplicate(pop,offspring['code']):
    #         if self.debug:
    #             print("duplicated code, wait 1 second and retrying ... ")
    #         time.sleep(1)
    #         p,offspring = self._get_alg(pop,operator)
    #     self.code2file(offspring['code'])
    #     try:
    #         fitness= self.interface_eval.evaluate()
    #     except:
    #         fitness = None
    #     offspring['objective'] =  fitness
    #     #offspring['other_inf'] =  first_gap
    #     while (fitness == None):
    #         if self.debug:
    #             print("warning! error code, retrying ... ")
    #         p,offspring = self._get_alg(pop,operator)
    #         while self.check_duplicate(pop,offspring['code']):
    #             if self.debug:
    #                 print("duplicated code, wait 1 second and retrying ... ")
    #             time.sleep(1)
    #             p,offspring = self._get_alg(pop,operator)
    #         self.code2file(offspring['code'])
    #         try:
    #             fitness= self.interface_eval.evaluate()
    #         except:
    #             fitness = None
    #         offspring['objective'] =  fitness
    #         #offspring['other_inf'] =  first_gap
    #     offspring['objective'] = np.round(offspring['objective'],5) 
    #     #offspring['other_inf'] = np.round(offspring['other_inf'],3)
    #     return p,offspring
