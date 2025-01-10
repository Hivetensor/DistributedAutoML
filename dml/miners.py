import json
import logging
import operator
import os
import queue
import random
import sys
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from multiprocessing import Queue, Process, Value
from uuid import UUID

import dill
import numpy as np
import requests
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from dml.chain.chain_manager import ChainManager
from dml.data import load_datasets
from dml.deap_individual import FitnessMax, Individual
from dml.models import BaselineNN, EvolvableNN, get_model_for_dataset, TorchEvolvedOptimizer
from dml.ops import create_pset
from dml.gp_fix import SafePrimitiveTree
from dml.destinations import (
    PushMixin,
    HFChainPushDestination, PoolPushDestination,
)
from dml.gene_io import save_individual_to_json, load_individual_from_json, safe_eval
from dml.gp_fix import SafePrimitiveTree
from dml.models import EvolvableNN, get_model_for_dataset
from dml.ops import create_pset
from dml.record import GeneRecordManager
from dml.utils import set_seed, compute_chain_hash

LOCAL_STORAGE_PATH = "./checkpoints"
os.makedirs(LOCAL_STORAGE_PATH, exist_ok=True)


class BaseMiner(ABC, PushMixin):
    def __init__(self, config):
        self.config = config
        self.device = self.config.device
        self.seed = self.config.Miner.seed
        set_seed(self.seed)
        self.setup_logging()
        self.metrics_file = config.metrics_file
        self.metrics_data = []
        self.push_destinations = []
        self.gene_record_manager = GeneRecordManager()

        # Push tracking
        self.last_push_attempt = 0  # Timestamp of last push attempt
        self.push_cooldown = 30 * 60  # 30 minutes in seconds
        self.last_push_success = False
        self.best_solution = {
            "individual": None,
            "fitness": float("-inf"),
            "pushed": False,
            "push_attempts": 0,
            "max_push_attempts": 3,  # Maximum number of push attempts
            "hash": None,
        }

        # Initialize record keeping
        self.push_record_file = os.path.join(LOCAL_STORAGE_PATH, "push_record.json")
        self._load_push_record()

        # DEAP utils
        self.initialize_deap()

    def _load_push_record(self):
        """Load the push record from disk if it exists"""
        try:
            if os.path.exists(self.push_record_file):
                with open(self.push_record_file, "r") as f:
                    record = json.load(f)
                    self.last_push_attempt = record.get("last_push_attempt", 0)
                    self.last_push_success = record.get("last_push_success", False)
                    # Best solution data will be reconstructed during mining
        except Exception as e:
            logging.warning(f"Failed to load push record: {e}")

    def _save_push_record(self):
        """Save the current push record to disk"""
        try:
            record = {
                "last_push_attempt": self.last_push_attempt,
                "last_push_success": self.last_push_success,
                "best_fitness": (
                    float(self.best_solution["fitness"])
                    if self.best_solution["individual"]
                    else float("-inf")
                ),
            }
            with open(self.push_record_file, "w") as f:
                json.dump(record, f)
        except Exception as e:
            logging.error(f"Failed to save push record: {e}")

    def should_attempt_push(self) -> bool:
        """Check if enough time has passed and we should attempt a push"""
        current_time = time.time()
        time_since_last_push = current_time - self.last_push_attempt

        # If last push failed and we haven't exceeded max attempts
        if (
                not self.last_push_success
                and self.best_solution["push_attempts"]
                < self.best_solution["max_push_attempts"]
        ):
            return True

        # If cooldown period has passed
        return time_since_last_push >= self.push_cooldown

    def attempt_push(self, individual, generation):
        """Attempt to push the solution with tracking"""
        try:
            current_time = time.time()
            commit_message = f"{generation}_{individual.fitness.values[0]:.4f}"

            # Attempt the push
            success = super().push_to_remote(individual, commit_message)

            if success:
                # Update tracking on success
                self.last_push_attempt = current_time
                self.last_push_success = True
                self.best_solution["pushed"] = True
                self.best_solution["push_attempts"] = 0  # Reset attempts counter
                logging.info(f"Successfully pushed solution at generation {generation}")
            else:
                raise MetadataError("Failed Push - Reason unknown")

        except MetadataError as e:
            # Update tracking on failure
            self.last_push_success = False
            self.best_solution["push_attempts"] += 1
            logging.error(f"Push attempt failed: {e}")

        except Exception as e:
            # Update tracking on failure
            self.last_push_success = False
            self.best_solution["push_attempts"] += 1
            logging.error(f"Push attempt failed: {e}")

        finally:
            self._save_push_record()

    def get_hash(self, individual):
        """Gets the hash of an individual's gene expression"""
        return compute_chain_hash(str(individual))

    def update_best_solution(self, individual, generation):
        """Update the best solution if new one is better"""
        current_fitness = individual.fitness.values[0]
        current_hash = self.get_hash(individual)

        if (current_fitness > self.best_solution["fitness"]) \
                and (current_hash != self.best_solution["hash"]):
            self.best_solution["individual"] = deepcopy(individual)
            self.best_solution["fitness"] = current_fitness
            self.best_solution["pushed"] = False
            self.best_solution["push_attempts"] = 0
            self.best_solution["hash"] = current_hash
            logging.info(
                f"New best solution found at generation {generation} with fitness {current_fitness:.4f}"
            )
            return True
        return False

    def initialize_deap(self):
        from dml.deap_individual import Individual

        self.toolbox = base.Toolbox()
        self.pset = create_pset()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=3)
        self.toolbox.register(
            "individual", tools.initIterate, Individual, self.toolbox.expr
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        self.toolbox.register("evaluate", self.create_n_evaluate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.decorate(
            "mate",
            gp.staticLimit(
                operator.attrgetter("height"), self.config.Miner.gp_tree_height
            ),
        )

        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register(
            "mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset
        )
        self.toolbox.decorate(
            "mutate",
            gp.staticLimit(
                operator.attrgetter("height"), self.config.Miner.gp_tree_height
            ),
        )


    def create_baseline_model(self):
        return BaselineNN(input_size=28 * 28, hidden_size=128, output_size=10).to(
            self.device
        )

    def measure_baseline(self, datasets):
        fitness = 0.0
        for dataset in datasets:
            model = self.create_baseline_model(dataset.name)
            try:
                self.train(model, train_loader=dataset.train_loader)
                fitness += (
                        self.evaluate(model, val_loader=dataset.val_loader) * dataset.weight
                )
            except Exception as e:
                logging.error(e)
                return (0.0,)
        logging.info(f"Baseline model accuracy: {fitness:.4f}")
        return (fitness,)

    def save_checkpoint(
            self,
            population,
            hof,
            best_individual_all_time,
            generation,
            random_state,
            torch_rng_state,
            numpy_rng_state,
            checkpoint_file,
    ):
        checkpoint = {
            "population": population,
            "hof": hof,
            "best_individual_all_time": best_individual_all_time,
            "generation": generation,
            "random_state": random_state,
            "torch_rng_state": torch_rng_state,
            "numpy_rng_state": numpy_rng_state,
        }
        with open(checkpoint_file, "wb") as cp_file:
            dill.dump(checkpoint, cp_file)

    def load_checkpoint(self, checkpoint_file):

        with open(checkpoint_file, "rb") as cp_file:
            checkpoint = dill.load(cp_file)

        # Restore population, hof, and best_individual_all_time directly
        population = checkpoint["population"]
        hof = checkpoint["hof"]
        best_individual_all_time = checkpoint["best_individual_all_time"]

        # Restore random states
        random.setstate(checkpoint["random_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])

        # Get the generation number
        generation = checkpoint["generation"]

        return population, hof, best_individual_all_time, generation

        # Reconstruct best_individual_all_time
        best_individual_str, best_individual_fitness = checkpoint[
            "best_individual_all_time"
        ]
        if best_individual_str is not None:
            best_individual_all_time = creator.Individual(
                SafePrimitiveTree.from_string(best_individual_str, self.pset, safe_eval)
            )
            best_individual_all_time.fitness.values = best_individual_fitness
        else:
            best_individual_all_time = None

        # Restore random states
        random_state = checkpoint["random_state"]
        torch_rng_state = checkpoint["torch_rng_state"]
        numpy_rng_state = checkpoint["numpy_rng_state"]

        # Set the random states
        random.setstate(random_state)
        torch.set_rng_state(torch_rng_state)
        np.random.set_state(numpy_rng_state)

        # Get the generation number
        generation = checkpoint["generation"]

        return population, hof, best_individual_all_time, generation

    # @abstractmethod
    # def load_data(self):
    #     pass

    @abstractmethod
    def create_model(self, genome):
        pass

    @abstractmethod
    def train(self, model, train_loader):
        pass

    @abstractmethod
    def evaluate(self, model, val_loader):
        pass

    def create_n_evaluate(self, individual, datasets):
        fitness = 0.0
        for dataset in datasets:
            for architecture in self.config.Miner.architectures[dataset.name]:

                model = self.create_model(individual, dataset.name, architecture)
                model[0].to(self.config.device)
                try:
                    self.train(model, train_loader=dataset.train_loader)
                    fitness += (
                        self.evaluate(model, val_loader=dataset.val_loader) * dataset.weight * self.config.Miner.architectures_weights[architecture]
                    )
                except Exception as e:
                    logging.error(e)
                    return (0.0,)

        return (fitness,)

    def log_mutated_child(self, offspring, generation):
        unpacked_code = self.unpacker.unpack_function_genome(offspring)
        log_filename = f"mutated_child_gen_{generation}.py"
        with open(log_filename, "w") as f:
            f.write(unpacked_code)
        logging.info(
            f"Logged mutated child for generation {generation} to {log_filename}"
        )

    def mine(self, initial_population=None):
        #self.measure_baseline()
        datasets = load_datasets(
            self.config.Miner.architectures.keys(), 
            batch_size=self.config.Miner.batch_size,
            seed=self.config.Miner.seed
        )

        checkpoint_file = os.path.join(LOCAL_STORAGE_PATH, "evolution_checkpoint.pkl")

        # Check if checkpoint exists
        if initial_population:
            population = []
            for func in initial_population:
                individual = creator.Individual(
                    SafePrimitiveTree.from_string(
                        func['function_code'], self.pset, safe_eval
                    )
                )
                population.append(individual)
        else:
            if os.path.exists(checkpoint_file):
                population, hof, best_individual_all_time, start_generation = self.load_checkpoint(checkpoint_file)
                if best_individual_all_time is not None:
                    self.best_solution["individual"] = best_individual_all_time
                    self.best_solution["fitness"] = best_individual_all_time.fitness.values[0]
                    self.best_solution["hash"] = self.get_hash(best_individual_all_time)
                logging.info(f"Resuming from generation {start_generation}")
            else:
                population = self.toolbox.population(n=self.config.Miner.population_size)
                hof = tools.HallOfFame(1)
                start_generation = 0
                logging.info("Starting evolution from scratch")

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)  # Replaced np with torch

        for generation in tqdm(range(start_generation, self.config.Miner.generations)):
            # Evaluate the entire population
            for i, ind in enumerate(population):
                if not ind.fitness.valid:
                    ind.fitness.values = self.toolbox.evaluate(ind, datasets)
                logging.debug(
                    f"Gen {generation}, Individual {i}: Fitness = {ind.fitness.values[0]}"
                )

            # Update best solution
            best_in_gen = tools.selBest(population, 1)[0]
            best_updated = self.update_best_solution(best_in_gen, generation)

            # Check if we should attempt a push
            if (best_updated or not self.last_push_success) and self.should_attempt_push():
                self.attempt_push(self.best_solution["individual"], generation)

            # Select the next generation individuals
            offspring = self.toolbox.select(population, len(population))

            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for i in range(0, len(offspring), 2):
                if random.random() < 0.5:
                    if i + 1 < len(offspring):
                        child1, child2 = offspring[i], offspring[i + 1]
                        safe_temp1, safe_temp2 = self.toolbox.clone(child1), self.toolbox.clone(child2)
                        self.toolbox.mate(child1, child2)

                        if child1.height > self.config.Miner.gp_tree_height:
                            offspring[i] = safe_temp1

                        if child2.height > self.config.Miner.gp_tree_height:
                            offspring[i + 1] = safe_temp2

                        del offspring[i].fitness.values
                        if i + 1 < len(offspring):
                            del offspring[i + 1].fitness.values

                        for i in range(len(offspring)):
                            if random.random() < 0.2:
                                mutant = self.toolbox.clone(offspring[i])
                                self.toolbox.mutate(mutant)
                                if mutant.height <= self.config.Miner.gp_tree_height:
                                    offspring[i] = mutant
                                    del offspring[i].fitness.values

            # Evaluate the new individuals
            # invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            # fitnesses = map(self.toolbox.evaluate, invalid_ind, [train_loader]*len(invalid_ind), [val_loader]*len(invalid_ind))
            # for ind, fit in zip(invalid_ind, fitnesses):
            #     ind.fitness.values = fit

            # Update the population
            population[:] = offspring

            # Update the hall of fame with the generated individuals
            hof.update(population)

            # Save checkpoint
            self.save_checkpoint(
                population,
                hof,
                self.best_solution["individual"],
                generation,
                random.getstate(),
                torch.get_rng_state(),
                np.random.get_state(),
                checkpoint_file,
            )

            height = [ind.height for ind in population if ind.height]

            if generation % self.config.Miner.check_registration_interval == 0:
                self.config.bittensor_network.sync()

            logging.info(
                f"Generation {generation}: Best fitness = {self.best_solution['fitness']:.4f}"
            )

        return self.best_solution["individual"]

    @staticmethod
    def setup_logging(log_file="miner.log"):
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


class BaseHuggingFaceMiner(BaseMiner):
    def __init__(self, config):
        super().__init__(config)
        self.push_destinations.append(
            HFChainPushDestination(
                repo_name=config.gene_repo,
                chain_manager=ChainManager(
                    config.bittensor_network.subtensor,
                    config.Bittensor.netuid,
                    config.bittensor_network.wallet,
                ),
                compute_hash_fn=lambda gene: self.gene_record_manager._compute_function_signature(
                    self.toolbox.compile(expr=gene)
                ),
                config=config
            )
        )


class BaseMiningPoolMiner(BaseMiner):
    def __init__(self, config):
        super().__init__(config)
        self.pool_destination = PoolPushDestination(
            config.Miner.pool_url,
            config.bittensor_network.wallet,
            config.Miner.miner_operation
        )
        self.miner_operation = config.Miner.miner_operation
        self.wallet = config.bittensor_network.wallet
        self.push_destinations.append(self.pool_destination)

    def evaluate_function(self, function_code: str) -> float:
        """
        Evaluate a loss function by testing it on multiple datasets/architectures
        Returns a normalized score between 0-1
        """
        try:
            # Safely compile the function code to get callable
            loss_function = self.toolbox.compile(function_code)

            total_score = 0.0
            datasets = load_datasets(
                self.config.Miner.architectures.keys(),
                batch_size=self.config.Miner.batch_size,
                seed=self.config.Miner.seed
            )

            # Test across each dataset and architecture combination
            for dataset in datasets:
                for architecture in self.config.Miner.architectures[dataset.name]:
                    model = get_model_for_dataset(dataset.name, architecture).to(self.device)
                    optimizer = torch.optim.Adam(model.parameters())

                    try:
                        # Training phase with the loss function
                        model.train()
                        for epoch in range(self.config.Miner.training_iterations):
                            for inputs, targets in dataset.train_loader:
                                inputs = inputs.to(self.device)
                                targets = targets.to(self.device)

                                optimizer.zero_grad()
                                outputs = model(inputs)
                                targets_one_hot = F.one_hot(targets, outputs.shape[-1]).float()

                                try:
                                    loss = loss_function(outputs, targets_one_hot)
                                    loss.backward()
                                    optimizer.step()
                                except Exception as e:
                                    logging.error(f"Loss computation failed: {e}")
                                    return 0.0

                        # Evaluation phase
                        score = self.evaluate(model, dataset.val_loader)
                        weighted_score = score * dataset.weight * self.config.Miner.architectures_weights[architecture]
                        total_score += weighted_score

                    except Exception as e:
                        logging.error(f"Evaluation failed for {dataset.name}/{architecture}: {e}")
                        continue

            # Normalize final score to 0-1 range
            total_architectures = sum(len(archs) for archs in self.config.Miner.architectures.values())
            final_score = total_score / total_architectures

            return min(max(final_score, 0.0), 1.0)  # Ensure score is between 0-1

        except Exception as e:
            logging.error(f"Function evaluation failed: {e}")
            return 0.0

    def mine(self, initial_population=None):
        while True:
            task = self.pool_destination.request_task(self.miner_operation)
            if not task:
                time.sleep(60)  # Wait before retrying
                continue

            if self.miner_operation == "evolve":
                evolved = super().mine(initial_population=task['functions'])  # Use existing evolution logic

                self.pool_destination.submit_result(
                    "evolve",
                    task["batch_id"],
                    {
                        "evolved_function": str(evolved),
                        "parent_functions": task["functions"]
                    }
                )
            else:
                for function in task["functions"]:
                    score = self.evaluate_function(function["function_code"])
                    self.pool_destination.submit_result(
                        "evaluate",
                        task["batch_id"],
                        {
                            "function_id": function["id"],
                            "score": score
                        }
                    )

            time.sleep(5)  # Prevent hammering the server


class IslandMiner(BaseMiner):
    def __init__(self, config):
        super().__init__(config)
        self.num_islands = config.Miner.num_processes
        self.migration_interval = config.Miner.migration_interval
        self.population_per_island = config.Miner.population_size // self.num_islands
        self.migrants_per_round = getattr(
            config.Miner, "migrants_per_round", 1
        )  # Default to 1 if not specified
        self._shutdown = False
        self.best_global_fitness = Value("d", -float("inf"))

        # Validate migrants per round
        if self.migrants_per_round >= self.population_per_island:
            logging.warning(
                f"migrants_per_round ({self.migrants_per_round}) must be less than population_per_island ({self.population_per_island})"
            )
            self.migrants_per_round = self.population_per_island // 2
            logging.warning(f"Setting migrants_per_round to {self.migrants_per_round}")

    def save_checkpoint(
            self,
            population,
            generation,
            local_best,
            local_best_fitness,
            random_state,
            torch_rng_state,
            numpy_rng_state,
            checkpoint_file,
    ):
        """Saves island checkpoint"""
        checkpoint = {
            "population": population,
            "generation": generation + 1,
            "local_best": local_best,
            "local_best_fitness": local_best_fitness,
            "random_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
        }
        with open(checkpoint_file, "wb") as f:
            dill.dump(checkpoint, f)

    def load_checkpoint(self, checkpoint_file):
        with open(checkpoint_file, "rb") as f:
            checkpoint = dill.load(f)
        population = checkpoint["population"]
        generation = checkpoint["generation"]
        local_best = checkpoint["local_best"]
        local_best_fitness = checkpoint["local_best_fitness"]
        # Restore random states
        random.setstate(checkpoint["random_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        return population, generation, local_best, local_best_fitness

    def run_island(
            self,
            island_id,
            migration_in_queue,
            migration_out_queue,
            stats_queue,
            global_best_queue,
    ):
        random.seed(self.seed + island_id)
        torch.manual_seed(self.seed + island_id)
        np.random.seed(self.seed + island_id)

        # Define checkpoint file path
        checkpoint_file = os.path.join(
            LOCAL_STORAGE_PATH, f"island_{island_id}_checkpoint.pkl"
        )

        # Check if checkpoint file exists and load it
        if os.path.exists(checkpoint_file):
            population, generation, local_best, local_best_fitness = (
                self.load_checkpoint(checkpoint_file)
            )
            logging.info(f"Island {island_id} resuming from generation {generation}")
        else:
            # Initialize population and variables
            logging.info("No checkpoint found, starting fresh")
            population = self.toolbox.population(n=self.population_per_island)
            generation = 0
            local_best = None
            local_best_fitness = -float("inf")
            logging.info(f"Island {island_id} starting from scratch")

        datasets = load_datasets(self.config.Miner.architectures.keys())

        while generation < self.config.Miner.generations:
            # Evolution step
            if self._shutdown:
                break
            offspring = algorithms.varOr(
                population,
                self.toolbox,
                lambda_=self.population_per_island,
                cxpb=0.5,
                mutpb=0.2,
            )
            # Evaluate
            for ind in offspring:
                if not ind.fitness.valid:
                    ind.fitness.values = self.create_n_evaluate(ind, datasets)

                    # Check for special migration case
                    if ind.fitness.values[0] > local_best_fitness:
                        local_best = deepcopy(ind)
                        local_best_fitness = ind.fitness.values[0]

            if (
                    local_best.fitness.values[0] > self.best_global_fitness.value
            ):  # Read is atomic
                global_best_queue.put((island_id, deepcopy(local_best)))
                logging.info(
                    f"Island {island_id} found potential global best: {local_best.fitness.values[0]:.4f}. fx: {str(local_best)}"
                )

            population = self.toolbox.select(offspring, self.population_per_island)

            # Collect stats
            fits = [ind.fitness.values[0] for ind in population]
            stats = {
                "island": island_id,
                "generation": generation,
                "best": np.max(fits),
                "avg": np.sum(fits) / len(fits),
                "worst": np.min(fits),
                "std": np.std(fits),  # torch.tensor(fits).std().item()
            }
            stats_queue.put(stats)

            # Regular Migration
            if generation % self.migration_interval == 0:
                # Select top N individuals to migrate
                migrants = [
                    deepcopy(ind)
                    for ind in tools.selBest(population, self.migrants_per_round)
                ]
                migration_out_queue.put((island_id, migrants))
                logging.info(
                    f"Island {island_id} sending {len(migrants)} migrants with fitness values: "
                    + ", ".join([f"{ind.fitness.values[0]:.4f}" for ind in migrants])
                )

                try:
                    # Receive migrants from another island
                    source_island, incoming_migrants = migration_in_queue.get_nowait()
                    logging.info(
                        f"Island {island_id} receiving {len(incoming_migrants)} migrants from Island {source_island} "
                        + "with fitness values: "
                        + ", ".join(
                            [
                                f"{ind.fitness.values[0]:.4f}"
                                for ind in incoming_migrants
                            ]
                        )
                    )

                    # Replace worst individuals with incoming migrants
                    worst_indices = [
                        population.index(ind)
                        for ind in tools.selWorst(population, len(incoming_migrants))
                    ]
                    for idx, migrant in zip(worst_indices, incoming_migrants):
                        population[idx] = migrant
                except queue.Empty:
                    logging.debug(f"Island {island_id} no incoming migrants this cycle")
                    pass

            # Save checkpoint
            random_state = random.getstate()
            torch_rng_state = torch.get_rng_state()
            numpy_rng_state = np.random.get_state()
            self.save_checkpoint(
                population,
                generation,
                local_best,
                local_best_fitness,
                random_state,
                torch_rng_state,
                numpy_rng_state,
                checkpoint_file,
            )
            logging.info(
                f"Island {island_id} saved checkpoint at generation {generation + 1}"
            )
            generation += 1

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
                    args=(
                        i,
                        migration_in_queues[i],
                        migration_out_queue,
                        stats_queue,
                        global_best_queue,
                    ),
                )
                processes.append(p)
                p.start()

            best_overall = None
            island_stats = {i: [] for i in range(self.num_islands)}
            generation = 0

            while any(p.is_alive() for p in processes):
                if generation > 0 and best_overall:
                    logging.info(
                        f"Best overall fitness:{best_overall.fitness.values[0]:.4f}"
                    )
                try:
                    candidates = []
                    try:
                        while True:
                            source_island, exceptional_ind = (
                                global_best_queue.get_nowait()
                            )
                            candidates.append((source_island, exceptional_ind))
                    except queue.Empty:
                        if candidates:
                            best_candidate = max(
                                candidates, key=lambda x: x[1].fitness.values[0]
                            )
                            source_island, candidate = best_candidate

                            if (
                                    best_overall is None
                                    or candidate.fitness.values[0]
                                    > best_overall.fitness.values[0]
                            ):
                                best_overall = deepcopy(candidate)
                                with self.best_global_fitness.get_lock():
                                    self.best_global_fitness.value = (
                                        candidate.fitness.values[0]
                                    )

                                self.push_to_remote(
                                    best_overall,
                                    f"{source_island}_{candidate.fitness.values[0]:.4f}",
                                )
                    except Exception as e:
                        logging.warning(f"Failed to push to remote: {e}")

                    # Handle regular migrations
                    try:
                        source_island, migrants = migration_out_queue.get_nowait()
                        for migrant in migrants:
                            if (
                                    best_overall is None
                                    or migrant.fitness.values[0]
                                    > best_overall.fitness.values[0]
                            ):
                                best_overall = deepcopy(migrant)
                                self.push_to_remote(
                                    best_overall,
                                    f"{generation}_{migrant.fitness.values[0]:.4f}",
                                )

                        for i, q in enumerate(migration_in_queues):
                            if i != source_island:
                                q.put((source_island, [deepcopy(m) for m in migrants]))
                    except queue.Empty:
                        pass

                    # Handle stats
                    try:
                        stats = stats_queue.get_nowait()
                        island_stats[stats["island"]].append(stats)
                        logging.info(
                            f"Island {stats['island']} Gen {stats['generation']}: "
                            f"Best={stats['best']:.4f} Avg={stats['avg']:.4f}"
                        )
                        generation = int(stats["generation"])
                    except queue.Empty:
                        time.sleep(0.1)

                    time.sleep(1)  # Adjust sleep time as needed

                except KeyboardInterrupt:
                    logging.info("Received interrupt signal, initiating shutdown...")
                    self.shutdown()
                    break

            for p in processes:
                p.join()

            return best_overall

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
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_data = datasets.MNIST(
            "../data", train=True, download=True, transform=transform
        )
        val_data = datasets.MNIST("../data", train=False, transform=transform)
        train_loader = DataLoader(
            train_data,
            batch_size=64,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        val_loader = DataLoader(
            val_data,
            batch_size=128,
            shuffle=False,
            generator=torch.Generator().manual_seed(self.seed),
        )
        return train_loader, val_loader

    def create_model(self, individual):
        set_seed(self.seed)
        return EvolvableNN(
            input_size=28 * 28,
            hidden_size=128,
            output_size=10,
            evolved_activation=self.toolbox.compile(expr=individual),
        ).to(self.device)

    def train(self, model, train_loader):
        set_seed(self.seed)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            if idx == self.config.Miner.training_iterations:
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
                if idx >= self.config.Miner.evaluation_iterations:
                    return correct / total
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return correct / total


class LossMiner(BaseMiner):
    def __init__(self, config):
        super().__init__(config)
        # self.seed_population()

    def load_data(self):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_data = datasets.MNIST(
            "../data", train=True, download=True, transform=transform
        )
        val_data = datasets.MNIST("../data", train=False, transform=transform)
        train_loader = DataLoader(
            train_data,
            batch_size=128,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        val_loader = DataLoader(
            val_data,
            batch_size=128,
            shuffle=False,
            generator=torch.Generator().manual_seed(self.seed),
        )
        return train_loader, val_loader

    def create_model(self, individual, dataset_name, architecture):
        set_seed(self.seed)
        return get_model_for_dataset(dataset_name, architecture), self.toolbox.compile(
            expr=individual
        )

    @staticmethod
    def safe_evaluate(func, outputs, labels):
        try:
            loss = func(outputs, labels)

            if loss is None:
                logging.error(f"Loss function returned None: {func}")
                return torch.tensor(float("inf"), device=outputs.device)

            if not torch.is_tensor(loss):
                logging.error(f"Loss function didn't return a tensor: {type(loss)}")
                return torch.tensor(float("inf"), device=outputs.device)

            if not torch.isfinite(loss).all():
                logging.warning(f"Non-finite loss detected: {loss}")
                return torch.tensor(float("inf"), device=outputs.device)

            if loss.ndim > 0:
                loss = loss.mean()

            return loss
        except Exception as e:
            logging.error(f"Error in loss calculation: {str(e)}")
            # logging.error(traceback.format_exc())
            return torch.tensor(float("inf"), device=outputs.device)

    def train(self, model_and_loss, train_loader):
        set_seed(self.seed)
        model, loss_function = model_and_loss
        optimizer = torch.optim.Adam(model.parameters())
        model.train()
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            if idx == self.config.Miner.training_iterations:
                break
            optimizer.zero_grad()
            outputs = model(inputs)
            targets_one_hot = torch.nn.functional.one_hot(
                targets, num_classes=outputs.shape[-1]
            ).float()
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
                if idx > self.config.Miner.evaluation_iterations:
                    break
                outputs = model(inputs)
                if len(outputs.shape) == 3:
                    _, predicted = outputs.max(dim=-1)
                    total += targets.numel()  # Count all elements
                    correct += predicted.eq(targets).sum().item()
                else:
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
        return correct / total

    def create_baseline_model(self, dataset_name):
        set_seed(self.seed)
        return get_model_for_dataset(dataset_name).to(self.device), torch.nn.MSELoss()


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
        population, logbook = algorithms.eaSimple(
            self.toolbox.population(n=self.config.Miner.population_size),
            self.toolbox,
            cxpb=0.5,
            mutpb=0.2,
            ngen=self.config.Miner.generations,
            stats=self.stats,
            halloffame=self.hof,
            verbose=True,
        )

        best_individual = self.hof[0]
        best_fitness = best_individual.fitness.values[0]

        logging.info(f"Best individual: {best_individual}")
        logging.info(f"Best fitness: {best_fitness}")

        return best_individual
    

class OptimizerMiner(BaseMiner):
    def __init__(self, config):
        super().__init__(config)
        self.eval_architectures = [
            {"input": 784, "hidden": 128, "output": 10},    # MNIST
            {"input": 3072, "hidden": 256, "output": 10},   # CIFAR
            {"vocab_size": 85, "embedding_dim": 384,        # Shakespeare
                "num_heads": 6, "num_layers": 6}
        ]

    def create_model(self, individual, dataset_name):
        set_seed(self.seed)
        compiled_func = self.toolbox.compile(expr=individual)
        model = get_model_for_dataset(dataset_name)
        return model, TorchEvolvedOptimizer(
            params=model.parameters(),
            evolved_func=compiled_func
        )


    def create_evolved_optimizer(self, individual):
        compiled_func = self.toolbox.compile(expr=individual)
        return TorchEvolvedOptimizer(
            params=self.current_model.parameters(),
            evolved_func=compiled_func
        )


    @staticmethod
    def safe_evaluate(func, outputs, labels):
        try:
            loss = func(outputs, labels)

            if loss is None:
                logging.error(f"Loss function returned None: {func}")
                return torch.tensor(float("inf"), device=outputs.device)

            if not torch.is_tensor(loss):
                logging.error(f"Loss function didn't return a tensor: {type(loss)}")
                return torch.tensor(float("inf"), device=outputs.device)

            if not torch.isfinite(loss).all():
                logging.warning(f"Non-finite loss detected: {loss}")
                return torch.tensor(float("inf"), device=outputs.device)

            if loss.ndim > 0:
                loss = loss.mean()

            return loss
        except Exception as e:
            logging.error(f"Error in loss calculation: {str(e)}")
            # logging.error(traceback.format_exc())
            return torch.tensor(float("inf"), device=outputs.device)

    def train(self, model_and_opt, train_loader):
        set_seed(self.seed)
        model, optimizer = model_and_opt
        
        model.train()
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            if idx == self.config.Miner.training_iterations:
                break
            optimizer.zero_grad()
            outputs = model(inputs)
            targets_one_hot = torch.nn.functional.one_hot(
                targets, num_classes=outputs.shape[-1]
            ).float()
            loss = F.cross_entropy(outputs, targets_one_hot)
            loss.backward()
            optimizer.step()

    def evaluate(self, model_and_opt, val_loader):
        set_seed(self.seed)
        model, _ = model_and_opt
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                if idx > self.config.Miner.evaluation_iterations:
                    break
                outputs = model(inputs)
                if len(outputs.shape) == 3:
                    _, predicted = outputs.max(dim=-1)
                    total += targets.numel()  # Count all elements
                    correct += predicted.eq(targets).sum().item()
                else:
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
        return correct / total

    
class ParallelOptimizerMiner(BaseMiner):
    def __init__(self):
        raise NotImplementedError


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

class OptimizerMinerPool(OptimizerMiner, BaseMiningPoolMiner):
    pass


class OptimizerMinerHF(OptimizerMiner, BaseHuggingFaceMiner):
    pass


class ParallelLossMinerPool(ParallelLossMiner, BaseMiningPoolMiner):
    pass


class ParallelLossMinerHF(ParallelLossMiner, BaseHuggingFaceMiner):
    pass



class ParallelOptimizerMinerPool(ParallelOptimizerMiner, BaseMiningPoolMiner):
    pass


class ParallelOptimizerMinerHF(ParallelOptimizerMiner, BaseHuggingFaceMiner):
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
        operation = config.Miner.miner_operation
        if platform == "pool":
            if not hasattr(config.Miner, "operation_type"):
                raise ValueError("When using pool, must specify operation_type ('evolve' or 'evaluate') in config")

            if operation not in ["evolve", "evaluate"]:
                raise ValueError("operation_type must be either 'evolve' or 'evaluate'")
            if miner_type == "activation":
                if core_count == 1:
                    return ActivationMinerPool(config)
                else:
                    return ParallelActivationMinerPool(config)
            elif miner_type == "loss":
                if core_count == 1:
                    return LossMinerPool(config)
                else:
                    return ParallelLossMinerPool(config)
            elif miner_type == "optimizer":
                if core_count == 1:
                    return OptimizerMinerPool(config)
                else:
                    return ParallelOptimizerMinerPool(config)
        elif platform == "hf":
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
            elif miner_type == "optimizer":
                if core_count == 1:
                    return OptimizerMinerHF(config)
                else:
                    return ParallelOptimizerMinerHF(config)

        raise ValueError(f"Unknown miner type: {miner_type} or platform: {platform}")

