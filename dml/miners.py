from abc import ABC, abstractmethod
from copy import deepcopy
from huggingface_hub import HfApi, Repository
import os
import requests
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import logging 
import pandas as pd
import numpy as np
import random 
import math 
import pickle
import operator
import torch.multiprocessing as torch_mp
import multiprocessing as mp
import time 

from multiprocessing import Queue, Process, Pool, Value
import queue

from deap import algorithms, base, creator, tools, gp
from functools import partial

from dml.models import BaselineNN, EvolvableNN
from dml.ops import create_pset
from dml.gene_io import save_individual_to_json, load_individual_from_json, safe_eval
from dml.gp_fix import SafePrimitiveTree
from dml.destinations import PushMixin, PoolPushDestination, HuggingFacePushDestination
from dml.utils import set_seed, calculate_tree_depth


LOCAL_STORAGE_PATH = "./checkpoints"
os.makedirs(LOCAL_STORAGE_PATH, exist_ok=True)

class BaseMiner(ABC, PushMixin):
    def __init__(self, config):
        self.config = config
        self.device = self.config.device
        self.seed = self.config.Miner.seed

        set_seed(self.seed)
    
        #self.migration_server_url = config.Miner.migration_server_url
        #self.migration_interval = config.Miner.migration_interval
        self.setup_logging()
        self.metrics_file = config.metrics_file
        self.metrics_data = []
        
        self.push_destinations = []

        
        # DEAP utils
        self.initialize_deap()

        
    def initialize_deap(self):
        self.toolbox = base.Toolbox()
        self.pset = create_pset()

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=3)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        self.toolbox.register("evaluate", self.create_n_evaluate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.decorate("mate", gp.staticLimit(operator.attrgetter('height'), self.config.Miner.gp_tree_height))
        
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)
        self.toolbox.decorate("mutate", gp.staticLimit(operator.attrgetter('height'), self.config.Miner.gp_tree_height))

    def log_metrics(self, metrics):
        self.metrics_data.append(metrics)
        
        # Save to CSV every 10 generations
        if len(self.metrics_data) % 10 == 0:
            df = pd.DataFrame(self.metrics_data)
            df.to_csv(self.metrics_file, index=False)

    

    def emigrate_genes(self, best_gene):
        
        # Submit best gene
        gene_data = []
        for gene in best_gene:
            gene_data.append(save_individual_to_json(gene=gene))

        response = requests.post(f"{self.migration_server_url}/submit_gene", json=gene_data)

        if response.status_code == 200:
            return True
        else:
            return False
        
    def immigrate_genes(self):
        # Get mixed genes from server
        response = requests.get(f"{self.migration_server_url}/get_mixed_genes")
        received_genes_data = response.json()

        if not self.migration_server_url:
            return []
        
        return [load_individual_from_json(gene_data=gene_data, function_decoder=self.function_decoder) 
                for gene_data in received_genes_data]

    #One migration cycle
    def migrate_genes(self,best_gene):
        self.emigrate_genes(best_gene)
        return self.immigrate_genes()
    
    def create_baseline_model(self):
        return BaselineNN(input_size=28*28, hidden_size=128, output_size=10).to(self.device)

    def measure_baseline(self):
        set_seed(self.seed)
        train_loader, val_loader = self.load_data()
        baseline_model = self.create_baseline_model()
        self.train(baseline_model, train_loader)
        self.baseline_accuracy = self.evaluate(baseline_model, val_loader)
        logging.info(f"Baseline model accuracy: {self.baseline_accuracy:.4f}")
    

    def save_checkpoint(self, population, hof, best_individual_all_time, generation, random_state, torch_rng_state, numpy_rng_state, checkpoint_file):
        # Convert population and hof individuals to string representations and save fitness
        population_data = [(str(ind), ind.fitness.values) for ind in population]
        hof_data = [(str(ind), ind.fitness.values) for ind in hof]
        
        # Serialize best_individual_all_time
        if best_individual_all_time is not None:
            best_individual_str = str(best_individual_all_time)
            best_individual_fitness = best_individual_all_time.fitness.values
        else:
            best_individual_str = None
            best_individual_fitness = None
        
        checkpoint = {
            'population': population_data,
            'hof': hof_data,
            'best_individual_all_time': (best_individual_str, best_individual_fitness),
            'generation': generation,
            'random_state': random_state,
            'torch_rng_state': torch_rng_state,
            'numpy_rng_state': numpy_rng_state
        }
        with open(checkpoint_file, 'wb') as cp_file:
            pickle.dump(checkpoint, cp_file)

    def load_checkpoint(self, checkpoint_file):
        with open(checkpoint_file, 'rb') as cp_file:
            checkpoint = pickle.load(cp_file)
        
        # Reconstruct population and hof individuals from strings and restore fitness
        population_data = checkpoint['population']
        population = []
        for expr_str, fitness_values in population_data:
            ind = creator.Individual(SafePrimitiveTree.from_string(expr_str, self.pset, safe_eval))
            ind.fitness.values = fitness_values
            population.append(ind)

        # population_depths = []
        # for member in population:
        #     depth = calculate_tree_depth(str(member))
        #     population_depths.append(depth)
        # breakpoint()
        
        hof_data = checkpoint['hof']
        hof = tools.HallOfFame(maxsize=len(hof_data))
        for expr_str, fitness_values in hof_data:
            ind = creator.Individual(SafePrimitiveTree.from_string(expr_str, self.pset, safe_eval))
            ind.fitness.values = fitness_values
            hof.insert(ind)
        
        
        
        # Reconstruct best_individual_all_time
        best_individual_str, best_individual_fitness = checkpoint['best_individual_all_time']
        if best_individual_str is not None:
            best_individual_all_time = creator.Individual(SafePrimitiveTree.from_string(best_individual_str, self.pset, safe_eval))
            best_individual_all_time.fitness.values = best_individual_fitness
        else:
            best_individual_all_time = None

        #print(calculate_tree_depth(str(best_individual_all_time)))
        #breakpoint()
        
        # Restore random states
        random_state = checkpoint['random_state']
        torch_rng_state = checkpoint['torch_rng_state']
        numpy_rng_state = checkpoint['numpy_rng_state']
        
        # Set the random states
        random.setstate(random_state)
        torch.set_rng_state(torch_rng_state)
        np.random.set_state(numpy_rng_state)
        
        # Get the generation number
        generation = checkpoint['generation']
        
        return population, hof, best_individual_all_time, generation


    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def create_model(self, genome):
        pass

    @abstractmethod
    def train(self, model, train_loader):
        pass

    @abstractmethod
    def evaluate(self, model, val_loader):
        pass

    
    def create_n_evaluate(self, individual, train_loader, val_loader):

        model = self.create_model(individual)
        try:
            self.train(model, train_loader=train_loader)
            fitness = self.evaluate(model, val_loader=val_loader)
        except:
            return 0.0,


        return fitness,

    def log_mutated_child(self, offspring, generation):
        unpacked_code = self.unpacker.unpack_function_genome(offspring)
        log_filename = f"mutated_child_gen_{generation}.py"
        with open(log_filename, 'w') as f:
            f.write(unpacked_code)
        logging.info(f"Logged mutated child for generation {generation} to {log_filename}")

    def mine(self):
        self.measure_baseline()
        train_loader, val_loader = self.load_data()
        
        checkpoint_file = os.path.join(LOCAL_STORAGE_PATH, 'evolution_checkpoint.pkl')
        
        # Check if checkpoint exists
        if os.path.exists(checkpoint_file):
            # Load checkpoint
            logging.info("Loading checkpoint...")
            population, hof, best_individual_all_time, start_generation = self.load_checkpoint(checkpoint_file)
            logging.info(f"Resuming from generation {start_generation}")
        else:
            # No checkpoint, start fresh
            population = self.toolbox.population(n=self.config.Miner.population_size)
            hof = tools.HallOfFame(1)
            best_individual_all_time = None
            start_generation = 0
            logging.info("Starting evolution from scratch")
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", torch.mean)
        stats.register("min", torch.min)  # Replaced np with torch
        
        for generation in tqdm(range(start_generation, self.config.Miner.generations)):
            # Evaluate the entire population
            for i, ind in enumerate(population):
                if not ind.fitness.valid:
                    ind.fitness.values = self.toolbox.evaluate(ind, train_loader, val_loader)
                logging.debug(f"Gen {generation}, Individual {i}: Fitness = {ind.fitness.values[0]}")

            height = []
            for ind in population:
                try:
                    height.append(ind.height)
                except:
                    pass
            
            print(f"max height pre: {max(height)}")

            # Select the next generation individuals
            offspring = self.toolbox.select(population, len(population))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))
        
            # Apply crossover and mutation on the offspring
            for i in range(0, len(offspring), 2):
                if random.random() < 0.5:
                    if i + 1 < len(offspring):
                        child1, child2 = offspring[i], offspring[i+1]
                        safe_temp1, safe_temp2 = self.toolbox.clone(child1), self.toolbox.clone(child2)
                        self.toolbox.mate(child1, child2)
                        
                        if child1.height > self.config.Miner.gp_tree_height:
                            offspring[i] = safe_temp1
                            
                        if child2.height > self.config.Miner.gp_tree_height:
                            offspring[i+1] = safe_temp2
                            

                        
                        del offspring[i].fitness.values
                        if i + 1 < len(offspring):
                            del offspring[i+1].fitness.values

                        for i in range(len(offspring)):
                            if random.random() < 0.2:
                                mutant = self.toolbox.clone(offspring[i])
                                self.toolbox.mutate(mutant)
                                if mutant.height <= self.config.Miner.gp_tree_height:
                                    offspring[i] = mutant
                                    del offspring[i].fitness.values

            # Evaluate the new individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind, [train_loader]*len(invalid_ind), [val_loader]*len(invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the population
            population[:] = offspring
        
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            for i, ind in enumerate(invalid_ind): 
                #try:
                ind.fitness.values = self.toolbox.evaluate(ind, train_loader, val_loader)
                # except SyntaxError as syn: 
                #     logging.warn(f"Syntax Error - Depth likely exceeded: {syn}")
                #     ind.fitness.values = [0]
                logging.debug(f"Gen {generation}, New Individual {i}: Fitness = {ind.fitness.values[0]}")
        
            # Update the hall of fame with the generated individuals
            hof.update(population)

            height = [ind.height for ind in population if ind.height]
            print(f"max height post: {max(height)}")
            
        
            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in population if not math.isinf(ind.fitness.values[0])]
            
            length = len(population)
            mean = sum(fits) / length if fits else float('inf')
            sum2 = sum(x*x for x in fits) if fits else float('inf')
            std = abs(sum2 / length - mean*2)*0.5 if fits else float('inf')
            
            logging.info(f"Generation {generation}: Valid fits: {len(fits)}/{length}")
            logging.info(f"Min {min(fits) if fits else float('inf')}")
            logging.info(f"Max {max(fits) if fits else float('inf')}")
            logging.info(f"Avg {mean}")
            logging.info(f"Std {std}")
        
            best_individual = tools.selBest(population, 1)[0]
        
            if best_individual_all_time is None:
                best_individual_all_time = best_individual
            if best_individual.fitness.values[0] > best_individual_all_time.fitness.values[0]:
                best_individual_all_time = deepcopy(best_individual)
                logging.info(f"New best gene found. Pushing to {self.config.gene_repo}")
                self.push_to_remote(best_individual_all_time, f"{generation}_{best_individual_all_time.fitness.values[0]:.4f})")
            
            if generation % self.config.Miner.check_registration_interval == 0:
                self.config.bittensor_network.sync()

            logging.info(f"Generation {generation}: Best accuracy = {best_individual.fitness.values[0]:.4f}")
            
            # Save checkpoint at the end of the generation
            random_state = random.getstate()
            torch_rng_state = torch.get_rng_state()
            numpy_rng_state = np.random.get_state()
            self.save_checkpoint(population, hof, best_individual_all_time, generation, random_state, torch_rng_state, numpy_rng_state, checkpoint_file)
        
        # Evolution finished
        logging.info("Evolution finished")
        
        # Remove the checkpoint file if evolution is complete
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        
        return best_individual_all_time

    
    # def evaluate_population(self, population, train_loader, val_loader):
    #     for genome in tqdm(population):
    #         genome.memory.reset()
    #         try:
    #             model = self.create_model(genome)
    #             self.train(model, train_loader)
    #             genome.fitness = self.evaluate(model, val_loader)
    #         except:
    #             genome.fitness = -9999

    @staticmethod
    def setup_logging(log_file='miner.log'):
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # def get_best_migrant(self):
    #     response = requests.get(f"{self.migration_server_url}/get_best_fitness")
    #     best_fitness = response.json()["best_fitness"]
    #     if type(best_fitness) == float:
    #         return best_fitness
    #     else:
    #         return -1

class BaseHuggingFaceMiner(BaseMiner):
    def __init__(self, config):
        super().__init__(config)
        self.push_destinations.append(HuggingFacePushDestination(config.gene_repo))

class BaseMiningPoolMiner(BaseMiner):
    def __init__(self, config):
        super().__init__(config)
        self.push_destinations.append(PoolPushDestination(config.Miner.pool_url, config.bittensor_network.wallet))
        self.pool_url = config.Miner.pool_url


    #TODO add a timestamp or sth to requests to prevent spoofing signatures
    def register_with_pool(self):
        data = self._prepare_request_data("register")
        response = requests.post(f"{self.pool_url}/register", json=data)
        return response.json()['success']

    def get_task_from_pool(self):
        data = self._prepare_request_data("get_task")
        response = requests.get(f"{self.pool_url}/get_task", json=data)
        return response.json()

    def submit_result_to_pool(self, best_genome):
        data = self._prepare_request_data("submit_result")
        data["result"] = save_individual_to_json(best_genome)
        response = requests.post(f"{self.pool_url}/submit_result", json=data)
        return response.json()['success']

    def get_rewards_from_pool(self):
        data = self._prepare_request_data("get_rewards")
        response = requests.get(f"{self.pool_url}/get_rewards", json=data)
        return response.json() 

    def update_config_with_task(self, task):
        # Update miner config with task-specific parameters if needed
        pass

class IslandMiner(BaseMiner):
    def __init__(self, config):
        super().__init__(config)
        self.num_islands = config.Miner.num_processes
        self.migration_interval = config.Miner.migration_interval
        self.population_per_island = config.Miner.population_size // self.num_islands
        self.migrants_per_round = getattr(config.Miner, 'migrants_per_round', 1)  # Default to 1 if not specified
        self._shutdown = False

        self.best_global_fitness = Value('d', -float('inf'))

        # Validate migrants per round
        if self.migrants_per_round >= self.population_per_island:
            logging.warning(f"migrants_per_round ({self.migrants_per_round}) must be less than population_per_island ({self.population_per_island})")
            self.migrants_per_round = self.population_per_island // 2
            logging.warning(f"Setting migrants_per_round to {self.migrants_per_round}")
        
    def run_island(self, island_id, migration_in_queue, migration_out_queue, stats_queue, global_best_queue):
        random.seed(self.seed + island_id)
        torch.manual_seed(self.seed + island_id)
        
        population = self.toolbox.population(n=self.population_per_island)
        train_loader, val_loader = self.load_data()
        
        local_best = None
        local_best_fitness = -float('inf')
        
        for generation in range(self.config.Miner.generations):
            # Evolution step
            if self._shutdown:
                break

            offspring = algorithms.varOr(
                population, self.toolbox,
                lambda_=self.population_per_island,
                cxpb=0.5, mutpb=0.2
            )
            
            # Evaluate
            for ind in offspring:
                if not ind.fitness.valid:
                    ind.fitness.values = self.create_n_evaluate(ind, train_loader, val_loader)
                    
                    # Check for special migration case
                    if ind.fitness.values[0] > local_best_fitness:
                        local_best = deepcopy(ind)
                        local_best_fitness = ind.fitness.values[0]
            
            if local_best.fitness.values[0] > self.best_global_fitness.value:  # Read is atomic
                global_best_queue.put((island_id, deepcopy(local_best)))
                logging.info(f"Island {island_id} found potential global best: {local_best.fitness.values[0]:.4f}. fx: {str(local_best)}")
            
            population = self.toolbox.select(offspring, self.population_per_island)
            
            # Collect stats
            fits = [ind.fitness.values[0] for ind in population]
            stats = {
                "island": island_id,
                "generation": generation,
                "best": max(fits),
                "avg": sum(fits) / len(fits),
                "worst": min(fits),
                "std": torch.tensor(fits).std().item()
            }
            stats_queue.put(stats)
            
            # Regular Migration
            if generation % self.migration_interval == 0:
                # Select top N individuals to migrate
                migrants = [deepcopy(ind) for ind in tools.selBest(population, self.migrants_per_round)]
                migration_out_queue.put((island_id, migrants))
                logging.info(f"Island {island_id} sending {len(migrants)} migrants with fitness values: " + 
                           ", ".join([f"{ind.fitness.values[0]:.4f}" for ind in migrants]))
                
                try:
                    # Receive migrants from another island
                    source_island, incoming_migrants = migration_in_queue.get_nowait()
                    logging.info(f"Island {island_id} receiving {len(incoming_migrants)} migrants from Island {source_island} " +
                               "with fitness values: " + ", ".join([f"{ind.fitness.values[0]:.4f}" for ind in incoming_migrants]))
                    
                    # Replace worst individuals with incoming migrants
                    worst_indices = [population.index(ind) for ind in tools.selWorst(population, len(incoming_migrants))]
                    for idx, migrant in zip(worst_indices, incoming_migrants):
                        population[idx] = migrant
                except:
                    logging.debug(f"Island {island_id} no incoming migrants this cycle")
                    pass

    def mine(self):
        try:
            migration_in_queues = [Queue() for _ in range(self.num_islands)]
            migration_out_queue = Queue()
            stats_queue = Queue()
            global_best_queue = Queue()
            
            processes = []
            for i in range(self.num_islands):
                p = Process(
                    target=self.run_island,
                    args=(i, migration_in_queues[i], migration_out_queue, stats_queue, global_best_queue)
                )
                processes.append(p)
                p.start()
            
            best_overall = None
            island_stats = {i: [] for i in range(self.num_islands)}
            generation = 0
            
            while any(p.is_alive() for p in processes):
                # Handle exceptional individuals first
                if generation > 0:
                    logging.info(f"Best overall fitness:{best_overall.fitness.values[0]}")
                try:
                    candidates = []
                    try:
                        while True:  # Collect all available candidates
                            source_island, exceptional_ind = global_best_queue.get_nowait()
                            candidates.append((source_island, exceptional_ind))
                            
                    except queue.Empty:
                        if candidates:  # If we found any candidates
                            # Find the best candidate from this batch
                            best_candidate = max(candidates, key=lambda x: x[1].fitness.values[0])
                            source_island, candidate = best_candidate
                            
                            # Check if it's better than our overall best
                            if (best_overall is None or 
                                candidate.fitness.values[0] > best_overall.fitness.values[0]):
                                
                                
                                best_overall = deepcopy(candidate)
                                with self.best_global_fitness.get_lock():
                                    self.best_global_fitness.value = candidate.fitness.values[0]
                                    
                                self.push_to_remote(best_overall, 
                                                f"{source_island}_{candidate.fitness.values[0]:.4f}")
                                            
                except Exception as e:
                    logging.warn(f"Failed to push to remote: {e}")
                    
                # Handle regular migrations
                try:
                    source_island, migrants = migration_out_queue.get_nowait()
                    # Check if any migrant is better than current best
                    for migrant in migrants:
                        if (best_overall is None or 
                            migrant.fitness.values[0] > best_overall.fitness.values[0]):
                            best_overall = deepcopy(migrant)
                            self.push_to_remote(best_overall, 
                                            f"{generation}_{migrant.fitness.values[0]:.4f}")
                    
                    # Distribute migrants to other islands
                    for i, q in enumerate(migration_in_queues):
                        if i != source_island:
                            q.put((source_island, [deepcopy(m) for m in migrants]))
                except:
                    pass

                # Handle stats
                try:
                    stats = stats_queue.get_nowait()
                    island_stats[stats["island"]].append(stats)
                    logging.info(f"Island {stats['island']} Gen {stats['generation']}: "
                        f"Best={stats['best']:.4f} Avg={stats['avg']:.4f}")
                    generation = int(stats['generation'])
                except:
                    time.sleep(0.1)
                    
                time.sleep(10)

            for p in processes:
                p.join()
                
            return best_overall
        
        except KeyboardInterrupt:
            logging.info("Received interrupt signal, initiating shutdown...")
            self.shutdown()
            raise
        
        finally:
            # Clean up processes
            self.shutdown()
            
            # Clear all queues
            for q in migration_in_queues:
                while not q.empty():
                    try:
                        q.get_nowait()
                    except:
                        pass
            
            for q in [migration_out_queue, stats_queue, global_best_queue]:
                while not q.empty():
                    try:
                        q.get_nowait()
                    except:
                        pass

            # Terminate and join all processes
            for p in processes:
                if p.is_alive():
                    p.terminate()
                p.join(timeout=5)
                
            logging.info("All processes cleaned up")

    def shutdown(self):
        """Signal all islands to shutdown gracefully"""
        self._shutdown = True
        logging.info("Shutdown signal sent to all islands")


class ActivationMiner(BaseMiner):

    def load_data(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
        val_data = datasets.MNIST('../data', train=False, transform=transform)
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(self.seed))
        val_loader = DataLoader(val_data, batch_size=128, shuffle=False, generator=torch.Generator().manual_seed(self.seed))
        return train_loader, val_loader

    def create_model(self, individual):
        set_seed(self.seed)
        return EvolvableNN(
            input_size=28*28, 
            hidden_size=128, 
            output_size=10, 
            evolved_activation=self.toolbox.compile(expr=individual)
        ).to(self.device)

    def train(self, model, train_loader):
        set_seed(self.seed)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            if idx == 1:
                break
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    def evaluate(self, model, val_loader):
        set_seed(self.seed)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                if idx > 10:
                    return correct/total
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return correct / total
    
class LossMiner(BaseMiner):
    def __init__(self, config):
        super().__init__(config)
        #self.seed_population()

    # def seed_population(self, mse = True):
    #     population = self.toolbox.population(n=50)
    #     mse_population = [seed_with_mse(len(ind), ind.memory, ind.function_decoder, 
    #                                     ind.input_addresses, ind.output_addresses) for ind in population]
    #     return mse_population

    def load_data(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
        val_data = datasets.MNIST('../data', train=False, transform=transform)
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(self.seed))
        val_loader = DataLoader(val_data, batch_size=128, shuffle=False, generator=torch.Generator().manual_seed(self.seed))
        return train_loader, val_loader

    def create_model(self, individual):
        set_seed(self.seed)
        return BaselineNN(input_size=28*28, hidden_size=128, output_size=10).to(self.device), self.toolbox.compile(expr=individual)

    @staticmethod
    def safe_evaluate(func, outputs, labels):
        try:
            loss = func(outputs, labels)
            
            if loss is None:
                logging.error(f"Loss function returned None: {func}")
                return torch.tensor(float('inf'), device=outputs.device)
            
            if not torch.is_tensor(loss):
                logging.error(f"Loss function didn't return a tensor: {type(loss)}")
                return torch.tensor(float('inf'), device=outputs.device)
            
            if not torch.isfinite(loss).all():
                logging.warning(f"Non-finite loss detected: {loss}")
                return torch.tensor(float('inf'), device=outputs.device)
            
            if loss.ndim > 0:
                loss = loss.mean()
            
            return loss
        except Exception as e:
            logging.error(f"Error in loss calculation: {str(e)}")
            #logging.error(traceback.format_exc())
            return torch.tensor(float('inf'), device=outputs.device)

    def train(self, model_and_loss, train_loader):
        set_seed(self.seed)
        model, loss_function = model_and_loss
        optimizer = torch.optim.Adam(model.parameters())
        model.train()
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            if idx == 2:
                break
            optimizer.zero_grad()
            outputs = model(inputs)
            targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=10).float()
            loss = self.safe_evaluate(loss_function, outputs, targets_one_hot)
            loss.backward()
            optimizer.step()

    def evaluate(self, model_and_loss, val_loader):
        set_seed(self.seed)
        model, _ = model_and_loss
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                if idx > 10:
                    break
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return correct / total

    def create_baseline_model(self):
        return BaselineNN(input_size=28*28, hidden_size=128, output_size=10).to(self.device), torch.nn.MSELoss()


class SimpleMiner(BaseMiner):
    def load_data(self):
        x_data = torch.linspace(0, 10, 100)
        y_data = self.target_function(x_data)
        return (x_data, y_data), None

    def create_model(self, individual):
        return self.toolbox.compile(expr=individual)

    def train(self, model, train_data):
        # No training needed for this simple case
        pass

    def evaluate(self, model, val_data):
        set_seed(self.seed)
        x_data, y_data = val_data
        try:
            predicted = model(x_data)
            mse = torch.mean((predicted - y_data) ** 2)
            return 1 / (1 + mse)  # Convert MSE to a fitness score (higher is better)
        except:
            return 0

    @staticmethod
    def target_function(x):
        return (x / 2) + 2

    def mine(self):
        train_data, _ = self.load_data()
        
        # Use DEAP's algorithms module for the evolutionary process
        population, logbook = algorithms.eaSimple(self.toolbox.population(n=self.config.Miner.population_size), self.toolbox, 
                                                  cxpb=0.5, mutpb=0.2, 
                                                  ngen=self.config.Miner.generations, 
                                                  stats=self.stats, halloffame=self.hof, 
                                                  verbose=True)

        best_individual = self.hof[0]
        best_fitness = best_individual.fitness.values[0]

        logging.info(f"Best individual: {best_individual}")
        logging.info(f"Best fitness: {best_fitness}")

        return best_individual
    
class ActivationMinerPool(ActivationMiner, BaseMiningPoolMiner):
    pass

class ActivationMinerHF(ActivationMiner, BaseHuggingFaceMiner):
    pass

class ParallelActivationMiner(ActivationMiner, IslandMiner):
    pass

class ParallelActivationMinerPool(ParallelActivationMiner, BaseMiningPoolMiner):
    pass

class ParallelActivationMinerHF(ParallelActivationMiner, BaseHuggingFaceMiner):
    pass

class ParallelLossMiner(LossMiner, IslandMiner):
    pass

class LossMinerPool(LossMiner, BaseMiningPoolMiner):
    pass

class LossMinerHF(LossMiner, BaseHuggingFaceMiner):
    pass

class ParallelLossMinerPool(ParallelLossMiner, BaseMiningPoolMiner):
    pass

class ParallelLossMinerHF(ParallelLossMiner, BaseHuggingFaceMiner):
    pass


class SimpleMinerPool(SimpleMiner, BaseMiningPoolMiner):
    pass

class SimpleMinerHF(SimpleMiner, BaseHuggingFaceMiner):
    pass

class MinerFactory:
    @staticmethod
    def get_miner(config):
        miner_type = config.Miner.miner_type
        platform = config.Miner.push_platform
        core_count = config.Miner.num_processes
        if platform == 'pool':
            if miner_type == "activation":
                if core_count == 1:
                    return ActivationMinerPool(config)
                else:
                    return ParallelActivationMinerPool(config)
            elif miner_type == "loss":
                if core_count == 1:
                    return LossMinerPool(config)
                else:
                    return ParallelActivationMinerPool(config)
        elif platform == 'hf':
            if miner_type == "activation":
                if core_count == 1:
                    return ActivationMinerHF(config)
                else:
                    return ParallelActivationMinerHF(config)
            elif miner_type == "loss":
                if core_count == 1:
                    return LossMinerHF(config)
                else:
                    return ParallelLossMinerHF(config)
        
        raise ValueError(f"Unknown miner type: {miner_type} or platform: {platform}")