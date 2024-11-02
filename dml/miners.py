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

from typing import Any, List, Tuple

from deap import algorithms, base, creator, tools, gp

from dml.data import DatasetSpec, get_imagenet_1k_loaders, get_shakespeare_loaders, get_mnist_loaders, get_cifar10_loaders
from dml.models import BaselineNN, EvolvableNN, get_model_for_dataset
from dml.ops import create_pset
from dml.gene_io import save_individual_to_json, load_individual_from_json, safe_eval
from dml.gp_fix import SafePrimitiveTree
from dml.destinations import PushMixin, PoolPushDestination, HuggingFacePushDestination
from dml.utils import set_seed

LOCAL_STORAGE_PATH = "./checkpoints"
os.makedirs(LOCAL_STORAGE_PATH, exist_ok=True)

class BaseMiner(ABC, PushMixin):
    def __init__(self, config):
        self.config = config
        self.device = self.config.device
        self.seed = self.config.Miner.seed
        self.datasets: List[DatasetSpec] = []

        set_seed(self.seed)
    
        #self.migration_server_url = config.Miner.migration_server_url
        #self.migration_interval = config.Miner.migration_interval
        self.setup_logging()
        self.metrics_file = config.metrics_file
        self.metrics_data = []
        
        self.push_destinations = []

        self.mutation_log_interval = config.Miner.mutation_log_interval  # Add this line to get the log interval from config
        
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

     
    def create_baseline_model(self):
        return BaselineNN(input_size=28*28, hidden_size=128, output_size=10).to(self.device)

    # def measure_baseline(self):
    #     set_seed(self.seed)
    #     train_loader, val_loader = self.load_data()
    #     baseline_model = self.create_baseline_model()
    #     self.train(baseline_model, train_loader)
    #     self.baseline_accuracy = self.evaluate(baseline_model, val_loader)
    #     logging.info(f"Baseline model accuracy: {self.baseline_accuracy:.4f}")
    

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

    
    def create_n_evaluate(self, individual, datasets: List[DatasetSpec]) -> Tuple[float]:
        total_weighted_fitness = 0.0
        total_weight = sum(dataset.weight for dataset in datasets)

        for dataset in datasets:
            model = self.create_model(individual, dataset)
            try:
                self.train(
                    model=model,
                    train_loader=dataset.train_loader,
                    learning_rate=dataset.learning_rate,
                    training_iterations=dataset.training_iterations
                )
                fitness = self.evaluate(model, dataset.val_loader)
                logging.info(f"Fitness: {fitness} for dataset: {dataset.name}")
                total_weighted_fitness += (fitness * dataset.weight)
            except Exception as e:
                logging.error(f"Error training on {dataset.name}: {str(e)}")
                return 0.0,

        avg_fitness = total_weighted_fitness / total_weight
        return avg_fitness,

    def log_mutated_child(self, offspring, generation):
        unpacked_code = self.unpacker.unpack_function_genome(offspring)
        log_filename = f"mutated_child_gen_{generation}.py"
        with open(log_filename, 'w') as f:
            f.write(unpacked_code)
        logging.info(f"Logged mutated child for generation {generation} to {log_filename}")

    def mine(self):
        #self.measure_baseline()
        
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
                    #logging.info(f"loss under evaluation: {str(ind)}")
                    ind.fitness.values = self.toolbox.evaluate(ind, self.datasets)
                logging.debug(f"Gen {generation}, Individual {i}: Fitness = {ind.fitness.values[0]}")
        
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
                        del offspring[i+1].fitness.values

            for i in range(len(offspring)):
                if random.random() < 0.2:
                    mutant = self.toolbox.clone(offspring[i])
                    self.toolbox.mutate(mutant)
                    if mutant.height <= self.config.Miner.gp_tree_height:
                        offspring[i] = mutant
                        del offspring[i].fitness.values

            population[:] = offspring
        
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            for i, ind in enumerate(invalid_ind):
                ##Add multiple datasets to toolbox.evaluate
                #logging.info(f"loss under evaluation: {str(ind)}")
                ind.fitness.values = self.toolbox.evaluate(ind, self.datasets)
                logging.debug(f"Gen {generation}, New Individual {i}: Fitness = {ind.fitness.values[0]}")
        
            # Update the hall of fame with the generated individuals
            hof.update(population)
        
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
                self.push_to_remote(best_individual_all_time, f"Best gene (Gen {generation}, Acc {best_individual_all_time.fitness.values[0]:.4f})")
            
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

    def train(self, model, train_loader, learning_rate, training_iterations, **kwargs):
        set_seed(self.seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            if idx == training_iterations:
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
        self.datasets = self.load_datasets()
        #self.seed_population()

    # def seed_population(self, mse = True):
    #     population = self.toolbox.population(n=50)
    #     mse_population = [seed_with_mse(len(ind), ind.memory, ind.function_decoder, 
    #                                     ind.input_addresses, ind.output_addresses) for ind in population]
    #     return mse_population

    def load_datasets(self) -> List[DatasetSpec]:
        mnist_train, mnist_val = get_mnist_loaders(
            batch_size=self.config.Miner.batch_size
        )
        cifar_train, cifar_val = get_cifar10_loaders(
            batch_size=self.config.Miner.batch_size
        )

        # imagenet_train, imagenet_val = get_imagenet_1k_loaders(
        # data_dir=self.config.imagenet_path,
        # batch_size=self.config.Miner.batch_size
        # )
    
        # Add Shakespeare
        shakespeare_train, shakespeare_val, vocab_size = get_shakespeare_loaders(
            #text_path=self.config.shakespeare_path,
            batch_size=self.config.Miner.batch_size
        )
        
        return [
            DatasetSpec(
                name="mnist",
                input_size=28*28,
                output_size=10,
                train_loader=mnist_train,
                val_loader=mnist_val
            ),
            DatasetSpec(
                name="cifar10",
                input_size=32*32*3,
                output_size=10,
                train_loader=cifar_train,
                val_loader=cifar_val
            ),
            # DatasetSpec(
            # name="imagenet",
            # input_size=(3, 224, 224),
            # output_size=1000,
            # train_loader=imagenet_train,
            # val_loader=imagenet_val,
            # learning_rate=0.1,  # Different LR for ImageNet
            # weight=2.0  # Give more weight to ImageNet
            # ),
            DatasetSpec(
                name="shakespeare",
                input_size=32,  # sequence length
                output_size=vocab_size,
                train_loader=shakespeare_train,
                val_loader=shakespeare_val,
                hidden_size=32,  # Transformer embed size
                learning_rate=3e-4  # Standard LR for transformers
            )
        ]

    # def create_model(self, individual):
    #     set_seed(self.seed)
    #     return BaselineNN(input_size=28*28, hidden_size=128, output_size=10).to(self.device), self.toolbox.compile(expr=individual)
    
    def create_model(self, genome: Any, dataset_spec: DatasetSpec) -> torch.nn.Module:
        """Create model based on genome and dataset specifications"""
        base_model = get_model_for_dataset(
            dataset_name=dataset_spec.name,
            hidden_size=dataset_spec.hidden_size
        ).to(self.device)
        return base_model, self.toolbox.compile(expr=genome)

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

    def train(self, model, train_loader, learning_rate, training_iterations,**kwargs):
        set_seed(self.seed)
        model, loss_function = model
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model.train()
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            if idx == training_iterations:
                break
            optimizer.zero_grad()
            outputs = model(inputs)
            targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=outputs.shape[-1]).float()
            loss = self.safe_evaluate(loss_function, outputs, targets_one_hot)

            loss.backward()
            optimizer.step()

    def evaluate(self, model, val_loader):
        set_seed(self.seed)
        model, _ = model
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
                _, predicted = outputs.max(-1)
                if len(outputs.shape) > 2:
                    #Make it top5
                    total += (outputs.size(0) * outputs.size(1))
                    correct += predicted.eq(targets).sum().item()
                else:
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

class LossMinerPool(LossMiner, BaseMiningPoolMiner):
    pass

class LossMinerHF(LossMiner, BaseHuggingFaceMiner):
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

        if platform == 'pool':
            if miner_type == "activation":
                return ActivationMinerPool(config)
            elif miner_type == "loss":
                return LossMinerPool(config)
        elif platform == 'hf':
            if miner_type == "activation":
                return ActivationMinerHF(config)
            elif miner_type == "loss":
                return LossMinerHF(config)
        
        raise ValueError(f"Unknown miner type: {miner_type} or platform: {platform}")