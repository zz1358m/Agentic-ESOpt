import numpy as np
import json
import random
import time
import os

from .eoh_interface_EC import InterfaceEC
# main class for eoh
class EOH:

    # initilization
    def __init__(self, paras, problem, select, manage, **kwargs):

        self.paras = paras
        self.prob = problem
        self.select = select
        self.manage = manage
        
        # LLM settings
        self.use_local_llm = paras.llm_use_local
        self.llm_local_url = paras.llm_local_url
        self.api_endpoint = paras.llm_api_endpoint  # currently only API2D + GPT
        self.api_key = paras.llm_api_key
        self.llm_model = paras.llm_model

        # ------------------ RZ: use local LLM ------------------
        # self.use_local_llm = kwargs.get('use_local_llm', False)
        # assert isinstance(self.use_local_llm, bool)
        # if self.use_local_llm:
        #     assert 'url' in kwargs, 'The keyword "url" should be provided when use_local_llm is True.'
        #     assert isinstance(kwargs.get('url'), str)
        #     self.url = kwargs.get('url')
        # -------------------------------------------------------

        # Experimental settings       
        self.pop_size = paras.ec_pop_size  # popopulation size, i.e., the number of algorithms in population
        self.n_pop = paras.ec_n_pop  # number of populations

        self.operators = paras.ec_operators
        self.operator_weights = paras.ec_operator_weights
        self.operator_attempts = max(1, int(getattr(paras, "ec_operator_attempts", 1)))
        if paras.ec_m > self.pop_size or paras.ec_m == 1:
            print("m should not be larger than pop size or smaller than 2, adjust it to m=2")
            paras.ec_m = 2
        self.m = paras.ec_m

        self.debug_mode = paras.exp_debug_mode  # if debug
        self.ndelay = 1  # default

        self.use_seed = paras.exp_use_seed
        self.seed_path = paras.exp_seed_path
        self.load_pop = paras.exp_use_continue
        self.load_pop_path = paras.exp_continue_path
        self.load_pop_id = paras.exp_continue_id

        self.output_path = paras.exp_output_path

        self.exp_n_proc = paras.exp_n_proc
        
        self.timeout = paras.eva_timeout

        self.use_numba = paras.eva_numba_decorator

        print("- EoH parameters loaded -")

        # Set a random seed
        random.seed(2024)

    # add new individual to population
    def add2pop(self, population, offspring):
        for off in offspring:
            if off.get('objective') is None:
                continue
            for ind in population:
                if ind.get('objective') == off.get('objective'):
                    if (self.debug_mode):
                        print("duplicated result, retrying ... ")
                    break
            else:
                population.append(off)

    @staticmethod
    def _finite_objective(individual):
        objective = individual.get('objective')
        if objective is None:
            return None
        try:
            objective = float(objective)
        except (TypeError, ValueError):
            return None
        return objective if np.isfinite(objective) else None

    def _run_sampling(self, interface_ec, time_start):
        """Run independent i1 samples, optionally updating the model with ES."""
        mode = str(getattr(self.paras, "ec_run_mode", "sample"))
        batch_size = max(1, int(getattr(self.paras, "sample_batch_size", self.pop_size)))
        if mode == "sample_es":
            generations = max(1, int(self.n_pop))
            total = batch_size * generations
        else:
            total = max(1, int(getattr(self.paras, "sample_total", batch_size)))
            generations = (total + batch_size - 1) // batch_size

        print(
            f"- {mode}: independent i1 sampling, total={total}, "
            f"batch_size={batch_size}, generations={generations} -",
            flush=True,
        )
        top_population = []
        samples_written = 0
        valid_samples = 0
        samples_path = os.path.join(self.output_path, "results", "samples.jsonl")
        start_generation = 0
        resume = mode == 'sample' and bool(getattr(self.paras, 'sample_resume', False))

        if resume:
            if not os.path.isfile(samples_path):
                raise FileNotFoundError(f"Cannot resume sample run without {samples_path}")
            previous_samples = []
            with open(samples_path) as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        previous_samples.append(json.loads(line))
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"Invalid JSON in {samples_path} at line {line_number}"
                        ) from error
            samples_written = len(previous_samples)
            if samples_written > total:
                raise ValueError(
                    f"Resume source already has {samples_written} samples, exceeding target {total}."
                )
            for individual in previous_samples:
                objective = self._finite_objective(individual)
                if objective is not None:
                    valid_samples += 1
                    top_population.append(individual)
            top_population.sort(key=lambda individual: float(individual['objective']))
            top_population = top_population[:batch_size]
            start_generation = (samples_written + batch_size - 1) // batch_size
            print(
                f"- resumed {samples_written}/{total} samples at generation "
                f"{start_generation}/{generations}; previous best="
                f"{None if not top_population else top_population[0]['objective']} -",
                flush=True,
            )
        else:
            # Start clean when a run id is reused; generation JSON files follow
            # the same overwrite behavior as the original EoH result files.
            with open(samples_path, 'w'):
                pass

        for generation in range(start_generation, generations):
            current_batch_size = min(batch_size, total - samples_written)
            interface_ec.set_generation_context(generation, generations)
            _, batch = interface_ec.get_algorithm(
                [],
                "i1",
                offspring_count=current_batch_size,
            )

            # Preserve failed generations as attempted samples too. This makes
            # the JSONL length exactly T (or population * generations).
            if len(batch) < current_batch_size:
                batch.extend({
                    'algorithm': None,
                    'code': None,
                    'objective': float('inf'),
                    'other_inf': {'error': 'missing sampling result'},
                } for _ in range(current_batch_size - len(batch)))
            elif len(batch) > current_batch_size:
                batch = batch[:current_batch_size]

            for batch_index, individual in enumerate(batch):
                sample_index = samples_written + batch_index + 1
                metadata = individual.get('other_inf')
                if not isinstance(metadata, dict):
                    metadata = {}
                metadata.update({
                    'sample_mode': mode,
                    'sample_index': sample_index,
                    'sample_generation': generation + 1,
                    'sample_batch_index': batch_index + 1,
                })
                individual['other_inf'] = metadata
                objective = self._finite_objective(individual)
                if objective is not None:
                    valid_samples += 1
                    top_population.append(individual)

            top_population.sort(key=lambda individual: float(individual['objective']))
            top_population = top_population[:batch_size]

            history_filename = os.path.join(
                self.output_path,
                "results",
                "history",
                f"sample_generation_{generation + 1}.json",
            )
            with open(history_filename, 'w') as handle:
                json.dump(batch, handle, indent=5)

            with open(samples_path, 'a') as handle:
                for individual in batch:
                    handle.write(json.dumps(individual) + "\n")

            population_filename = os.path.join(
                self.output_path,
                "results",
                "pops",
                f"population_generation_{generation + 1}.json",
            )
            with open(population_filename, 'w') as handle:
                json.dump(top_population, handle, indent=5)

            if top_population:
                best_filename = os.path.join(
                    self.output_path,
                    "results",
                    "pops_best",
                    f"population_generation_{generation + 1}.json",
                )
                with open(best_filename, 'w') as handle:
                    json.dump(top_population[0], handle, indent=5)

            samples_written += current_batch_size
            best = None if not top_population else top_population[0]['objective']
            print(
                f"--- {generation + 1} of {generations} sampling generations finished. "
                f"Samples: {samples_written}/{total}; valid={valid_samples}; "
                f"best={best}; Time Cost: {((time.time() - time_start) / 60):.1f} m",
                flush=True,
            )

        summary = {
            'mode': mode,
            'operator': 'i1',
            'total_samples': samples_written,
            'valid_samples': valid_samples,
            'batch_size': batch_size,
            'generations': generations,
            'best_objective': None if not top_population else float(top_population[0]['objective']),
        }
        if mode == 'sample_es':
            summary['reward'] = 'negative_training_objective'
            summary['reward_normalization'] = getattr(
                self.paras, 'llm_es_reward_normalization', 'zscore'
            )
            summary['es_updates'] = generations
            invalid_reward_strategy = str(
                getattr(self.paras, 'llm_es_invalid_reward_strategy', 'current')
            ).strip().lower()
            summary['invalid_reward_strategy'] = (
                'valid_only_zscore_invalid_zero'
                if invalid_reward_strategy == 'zero'
                else 'batch_relative_below_worst_valid'
                if getattr(self.paras, 'llm_es_batch_relative_invalid_reward', False)
                else 'fixed_floor'
            )
        with open(os.path.join(self.output_path, "results", "sample_summary.json"), 'w') as handle:
            json.dump(summary, handle, indent=2)
    

    # run eoh 
    def run(self):

        print("- Evolution Start -")

        time_start = time.time()

        # interface for large language model (llm)
        # interface_llm = PromptLLMs(self.api_endpoint,self.api_key,self.llm_model,self.debug_mode)

        # interface for evaluation
        interface_prob = self.prob

        # interface for ec operators
        interface_ec = InterfaceEC(self.pop_size, self.m, self.api_endpoint, self.api_key, self.llm_model, self.use_local_llm, self.llm_local_url,
                                   self.debug_mode, interface_prob, select=self.select,n_p=self.exp_n_proc,
                                   timeout = self.timeout, use_numba=self.use_numba, paras=self.paras
                                   )

        if str(getattr(self.paras, "ec_run_mode", "eoh")) in {"sample", "sample_es"}:
            self._run_sampling(interface_ec, time_start)
            return

        # initialization
        population = []
        if self.use_seed:
            with open(self.seed_path) as file:
                data = json.load(file)
            population = interface_ec.population_generation_seed(data)
            filename = self.output_path + "/results/pops/population_generation_0.json"
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)
            n_start = 0
        else:
            if self.load_pop:  # load population from files
                print("load initial population from " + self.load_pop_path)
                with open(self.load_pop_path) as file:
                    data = json.load(file)
                for individual in data:
                    population.append(individual)
                print("initial population has been loaded!")
                n_start = self.load_pop_id
            else:  # create new population
                print("creating initial population:")
                population = interface_ec.population_generation()
                population = self.manage.population_management(population, self.pop_size)

                # print(len(population))
                # if len(population)<self.pop_size:
                #     for op in [self.operators[0],self.operators[2]]:
                #         _,new_ind = interface_ec.get_algorithm(population, op)
                #         self.add2pop(population, new_ind)
                #         population = self.manage.population_management(population, self.pop_size)
                #         if len(population) >= self.pop_size:
                #             break
                #         print(len(population))
     
                
                print(f"Pop initial: ")
                for off in population:
                    print(" Obj: ", off['objective'], end="|")
                print()
                print("initial population has been created!")
                # Save population to a file
                filename = self.output_path + "/results/pops/population_generation_0.json"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, 'w') as f:
                    json.dump(population, f, indent=5)
                n_start = 0

        # main loop
        n_op = len(self.operators)

        for pop in range(n_start, self.n_pop):  
            interface_ec.set_generation_context(pop, self.n_pop)
            #print(f" [{na + 1} / {self.pop_size}] ", end="|")         
            for i in range(n_op):
                op = self.operators[i]
                print(f" OP: {op}, [{i + 1} / {n_op}] ", end="|") 
                op_w = self.operator_weights[i]
                for attempt in range(self.operator_attempts):
                    if np.random.rand() < op_w:
                        parents, offsprings = interface_ec.get_algorithm(population, op)
                        self.add2pop(population, offsprings)  # Check duplication, and add the new offspring
                        for off in offsprings:
                            print(" Obj: ", off['objective'], end="|")
                # if is_add:
                #     data = {}
                #     for i in range(len(parents)):
                #         data[f"parent{i + 1}"] = parents[i]
                #     data["offspring"] = offspring
                #     with open(self.output_path + "/results/history/pop_" + str(pop + 1) + "_" + str(
                #             na) + "_" + op + ".json", "w") as file:
                #         json.dump(data, file, indent=5)
                # populatin management
                size_act = min(len(population), self.pop_size)
                population = self.manage.population_management(population, size_act)
                print()


            # Save population to a file
            filename = self.output_path + "/results/pops/population_generation_" + str(pop + 1) + ".json"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)

            # Save the best one to a file
            if len(population) > 0:
                filename = self.output_path + "/results/pops_best/population_generation_" + str(pop + 1) + ".json"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, 'w') as f:
                    json.dump(population[0], f, indent=5)
            else:
                print("Warning: empty population after management; skip best population save.")


            print(f"--- {pop + 1} of {self.n_pop} populations finished. Time Cost:  {((time.time()-time_start)/60):.1f} m")
            print("Pop Objs: ", end=" ")
            for i in range(len(population)):
                print(str(population[i]['objective']) + " ", end="")
            print()
