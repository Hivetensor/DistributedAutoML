import heapq
import logging
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from abc import ABC, abstractmethod
import time
from huggingface_hub import HfApi, Repository
import math 

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
from datasets import load_dataset
from PIL import Image

from typing import Any, Dict, Optional

from deap import algorithms, base, creator, tools, gp

from dml.models import BaselineNN, EvolvableNN, EvolvedLoss, ModelFactory
from dml.gene_io import load_individual_from_json
from dml.ops import create_pset_validator
from dml.record import GeneRecordManager
from dml.utils import set_seed



class BaseValidator(ABC):
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.chain_manager = config.chain_manager
        self.bittensor_network = config.bittensor_network
        self.interval = config.Validator.validation_interval
        self.gene_record_manager = GeneRecordManager()
        self.scores = {}
        self.normalized_scores = {}
        self.metrics_file = config.metrics_file
        self.metrics_data = []
        self.seed = self.config.Validator.seed

        self.penalty_factor = config.Validator.time_penalty_factor
        self.penalty_max_time = config.Validator.time_penalty_max_time

        set_seed(self.seed)
        
        # Initialize DEAP
        self.initialize_deap()

    def initialize_deap(self):
        self.toolbox = base.Toolbox()
        self.pset = create_pset_validator()

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=3)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)

    def log_metrics(self, iteration, scores):
        self.metrics_data.append({'iteration': iteration, **scores})
        
        if len(self.metrics_data) % 5 == 0:
            df = pd.DataFrame(self.metrics_data)
            df.to_csv(self.metrics_file, index=False)

    def calculate_time_penalty(self, new_timestamp: float, old_timestamp: float) -> float:
        time_diff = new_timestamp - old_timestamp
        if time_diff <= 0:
            return 1.0
        penalty = 1.0 - (time_diff / self.penalty_max_time) * self.penalty_factor
        return max(penalty, 1.0 - self.penalty_factor)
    
    def find_best_gene(self) -> Optional[Dict[str, Any]]:
        all_records = self.gene_record_manager.get_all_records()
        if not all_records:
            return None
        return max(all_records.values(), key=lambda x: x['performance'])

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def create_model(self, individual):
        pass

    @abstractmethod
    def evaluate(self, model, val_loader):
        pass

    def evaluate_individual(self, individual, val_loader):
        model = self.create_model(individual)
        return self.evaluate(model, val_loader),

    def create_baseline_model(self):
        return BaselineNN(input_size=28*28, hidden_size=128, output_size=10)

    def measure_baseline(self):
        _, val_loader = self.load_data()
        baseline_model = self.create_baseline_model()
        self.base_accuracy = self.evaluate(baseline_model, val_loader)
        logging.info(f"Baseline model accuracy: {self.base_accuracy:.4f}")

    def validate_and_score(self):
        set_seed(self.seed)

        logging.info("Receiving genes from chain")
        self.bittensor_network.sync(lite=True)
        
        if not self.check_registration():
            logging.info("This validator is no longer registered on the chain.")
            return

        _, val_loader = self.load_data()
        total_scores = 0
        best_gene = self.find_best_gene()
        current_time = time.time()

        for uid, hotkey_address in enumerate(self.bittensor_network.metagraph.hotkeys):

            hf_repo = self.chain_manager.retrieve_hf_repo(hotkey_address)
            remote_gene_hash = self.get_remote_gene_hash(hf_repo)
            if self.gene_record_manager.should_download(hotkey_address, remote_gene_hash):
                gene = self.receive_gene_from_hf(hf_repo)

                if gene is not None:
                    logging.info(f"Receiving gene from: {hotkey_address} ---> {hf_repo}")
                    
                    accuracy = self.evaluate_individual(gene[0], val_loader)[0]
                    accuracy_score = accuracy#max(0, accuracy - self.base_accuracy)

                    if best_gene is None or accuracy_score > best_gene['performance']:
                        final_score = accuracy_score
                        logging.info("No penalty applied.")
                    else:
                        time_penalty = self.calculate_time_penalty(current_time, best_gene['timestamp'])
                        final_score = accuracy_score * time_penalty
                        logging.info(f"Penalty applied. Original score: {accuracy_score:.4f}, Final score: {final_score:.4f}")

                    self.gene_record_manager.add_record(hotkey_address, remote_gene_hash, current_time, accuracy_score)

                    self.scores[hotkey_address] = final_score
                    logging.info(f"Accuracy: {accuracy:.4f}")
                    logging.info(f"Accuracy Score: {accuracy_score:.4f}")
                    
                else:
                    logging.info(f"No gene received from: {hotkey_address}")
                    self.scores[hotkey_address] = 0

            else:
                existing_record = self.gene_record_manager.get_record(hotkey_address)
                if existing_record:
                    time_penalty = self.calculate_time_penalty(existing_record['timestamp'], best_gene['timestamp'])
                    self.scores[hotkey_address] = existing_record['performance'] * time_penalty
                    logging.info(f"No new gene from: {hotkey_address}. Using existing score: {existing_record['performance']:.4f}")
                else:
                    self.scores[hotkey_address] = 0
                    logging.info(f"No record found for: {hotkey_address}")


        top_k = self.config.Validator.top_k
        top_k_weights = self.config.Validator.top_k_weight
        min_score = self.config.Validator.min_score

        # Define fixed weights for top-k miners

        score_hotkey_pairs = [(score, hotkey) for hotkey, score in self.scores.items() if score > 0]

        # Handle edge cases
        if not score_hotkey_pairs:
            # All scores are 0
            equal_weight = 1.0 / len(self.bittensor_network.metagraph.hotkeys)
            self.normalized_scores = {hotkey: equal_weight for hotkey in self.bittensor_network.metagraph.hotkeys}
        else:
            if top_k <= len(score_hotkey_pairs):
                top_k_scores = heapq.nlargest(top_k, score_hotkey_pairs)
                active_weights = top_k_weights[:len(top_k_scores)]
            else:
                top_k_scores = heapq.nlargest(len(score_hotkey_pairs), score_hotkey_pairs)
                active_weights = [1.0 / len(score_hotkey_pairs)] * len(score_hotkey_pairs)
            
            remaining_weight = 1.0 - sum(active_weights)
            weight_per_remaining = remaining_weight / (len(self.bittensor_network.metagraph.hotkeys) - len(top_k_scores))
                
            for i, (_, hotkey) in enumerate(top_k_scores):
                self.normalized_scores[hotkey] = active_weights[i]
            
            for hotkey in self.bittensor_network.metagraph.hotkeys:
                if hotkey not in self.normalized_scores:
                    self.normalized_scores[hotkey] = weight_per_remaining

        logging.info(f"Pre-normalization scores: {self.scores}")
        logging.info(f"Normalized scores: {self.normalized_scores}")
                
        self.log_metrics(len(self.metrics_data), self.normalized_scores)

        if self.bittensor_network.should_set_weights():
            self.bittensor_network.set_weights(self.normalized_scores)
            logging.info("Weights Setting attempted !")

    def check_registration(self):
        try:
            return self.bittensor_network.subtensor.is_hotkey_registered(
                netuid=self.bittensor_network.metagraph.netuid,
                hotkey_ss58=self.bittensor_network.wallet.hotkey.ss58_address
            )
        except:
            logging.warning("Failed to check registration, assuming still registered")
            return True

    def receive_gene_from_hf(self, repo_name):
        api = HfApi()
        try:
            file_info = api.list_repo_files(repo_id=repo_name)
            if "best_gene.json" in file_info:
                file_details = [thing for thing in api.list_repo_tree(repo_id=repo_name) if thing.path=="best_gene.json"]
                if file_details:
                    file_size = file_details[0].size
                    max_size = self.config.Validator.max_gene_size
                    
                    if file_size > max_size:
                        logging.warning(f"Gene file size ({file_size} bytes) exceeds limit ({max_size} bytes). Skipping download.")
                        return None
                    
                    gene_path = api.hf_hub_download(repo_id=repo_name, filename="best_gene.json")
                    gene_content = load_individual_from_json(pset=self.pset, toolbox=self.toolbox, filename=gene_path)
                    os.remove(gene_path)
                    return gene_content
                else:
                    logging.warning("Could not retrieve file details for best_gene.json")
            else:
                logging.info("best_gene.json not found in the repository")
        except Exception as e:
            logging.info(f"Error retrieving gene from Hugging Face: {str(e)}")
        return None
    
    def get_remote_gene_hash(self, repo_name: str) -> str:
        api = HfApi()
        try:
            file_info = api.list_repo_files(repo_id=repo_name)
            if "best_gene.json" in file_info:
                file_details = [thing for thing in api.list_repo_tree(repo_id=repo_name) if thing.path=="best_gene.json"]
                if file_details:
                    return file_details[0].blob_id  # This is effectively a hash of the file content
        except Exception as e:
            logging.error(f"Error retrieving gene hash from Hugging Face: {str(e)}")
        return ""  # Return empty string if we couldn't get the hash

    def start_periodic_validation(self):
        while True:
            self.validate_and_score()
            logging.info(f"One round done, sleeping for: {self.interval}")
            time.sleep(self.interval)

class DatasetMixin:
    """Mixin class that provides multi-dataset functionality"""
    def initialize_datasets(self) -> Dict[str, Dict[str, Any]]:
        datasets = {}
        
        # MNIST
        datasets['mnist'] = {
            'loader': self.load_mnist_data,
            'model_creator': lambda: ModelFactory.create_model('mnist', self.evolved_function).to(self.device),
            'input_size': 28 * 28,
            'output_size': 10,
            'weight': self.dataset_weights.get('mnist', 0.2)
        }
        
        # CIFAR-10
        datasets['cifar10'] = {
            'loader': self.load_cifar10_data,
            'model_creator': lambda: ModelFactory.create_model('cifar10', self.evolved_function).to(self.device),
            'input_size': 32 * 32 * 3,
            'output_size': 10,
            'weight': self.dataset_weights.get('cifar10', 0.3)
        }
        
        # Shakespeare
        datasets['shakespeare'] = {
            'loader': self.load_shakespeare_data,
            'model_creator': lambda: ModelFactory.create_model(
                'shakespeare', 
                self.evolved_function,
                vocab_size=50257, 
                embed_size=256
            ).to(self.device),
            'input_size': 256,
            'output_size': 50257,
            'weight': self.dataset_weights.get('shakespeare', 0.3)
        }
        
        # ImageNet-1k subset
        datasets['imagenet_subset'] = {
            'loader': self.load_imagenet_subset_data,
            'model_creator': lambda: ModelFactory.create_model('imagenet', self.evolved_function).to(self.device),
            'input_size': 224 * 224 * 3,
            'output_size': 1000,
            'weight': self.dataset_weights.get('imagenet_subset', 0.2)
        }
        
        return datasets

    def load_mnist_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
        val_data = datasets.MNIST('../data', train=False, transform=transform)
        return self._create_dataloaders(train_data, val_data)

    def load_cifar10_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_data = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
        val_data = datasets.CIFAR10('../data', train=False, transform=transform)
        return self._create_dataloaders(train_data, val_data)

    def load_shakespeare_data(self):
        dataset = load_dataset("tiny_shakespeare")
        
        def tokenize_function(examples):
            return self.tokenizer(examples['text'], truncation=True, max_length=128)
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        train_data = tokenized_dataset['train']
        val_data = tokenized_dataset['validation']
        
        return self._create_dataloaders(train_data, val_data, batch_size=16)

    def load_imagenet_subset_data(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Using a smaller subset of ImageNet for validation
        dataset = load_dataset("imagenet-1k", split="validation[:1000]")
        
        def preprocess_image(example):
            image = Image.open(example['image'].convert('RGB'))
            return {'image': transform(image), 'label': example['label']}
        
        processed_dataset = dataset.map(preprocess_image)
        train_size = int(0.8 * len(processed_dataset))
        train_data = processed_dataset.select(range(train_size))
        val_data = processed_dataset.select(range(train_size, len(processed_dataset)))
        
        return self._create_dataloaders(train_data, val_data)

    def _create_dataloaders(self, train_data, val_data, batch_size=None):
        if batch_size is None:
            batch_size = self.config.Validator.batch_size
        
        train_loader = DataLoader(
            train_data, 
            batch_size=batch_size, 
            shuffle=True, 
            generator=torch.Generator().manual_seed(self.seed)
        )
        val_loader = DataLoader(
            val_data, 
            batch_size=batch_size, 
            shuffle=False, 
            generator=torch.Generator().manual_seed(self.seed)
        )
        return train_loader, val_loader



class ActivationValidator(BaseValidator, DatasetMixin):

    def __init__(self, config):
        super().__init__(config)
        self.dataset_weights = config.Validator.dataset_weights
        self.datasets = self.initialize_datasets()

    def evaluate(self, individual, _):
        """
        Evaluates an individual across multiple datasets
        Args:
            individual: The evolved activation function
            _: Ignored (kept for compatibility)
        """
        self.evolved_function = self.toolbox.compile(expr=individual)
        total_score = 0.0
        
        for dataset_name, dataset_info in self.datasets.items():
            try:
                _, val_loader = dataset_info['loader']()
                model = dataset_info['model_creator']()
                score = self.evaluate_single_dataset(model, val_loader)
                weighted_score = score * dataset_info['weight']
                total_score += weighted_score
                
                logging.info(f"{dataset_name} Score: {score:.4f} (Weighted: {weighted_score:.4f})")
            except Exception as e:
                logging.error(f"Error evaluating {dataset_name}: {str(e)}")
                continue
        
        return total_score

    def evaluate_single_dataset(self, model, val_loader):
        """Evaluation logic for a single dataset"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return correct / total if total > 0 else 0.0
    
class LossValidator(BaseValidator, DatasetMixin):

    def __init__(self, config):
        super().__init__(config)
        self.dataset_weights = config.Validator.dataset_weights
        self.datasets = self.initialize_datasets()

    def evaluate(self, individual, _):
        """
        Evaluates a loss function individual across multiple datasets
        Args:
            individual: The evolved loss function
            _: Ignored (kept for compatibility)
        """
        self.evolved_function = self.toolbox.compile(expr=individual)
        total_score = 0.0
        
        for dataset_name, dataset_info in self.datasets.items():
            try:
                train_loader, val_loader = dataset_info['loader']()
                model = dataset_info['model_creator']()
                
                # Train with evolved loss
                self.train_model(model, train_loader, self.evolved_function)
                
                # Evaluate
                score = self.evaluate_single_dataset(model, val_loader)
                weighted_score = score * dataset_info['weight']
                total_score += weighted_score
                
                logging.info(f"{dataset_name} Score: {score:.4f} (Weighted: {weighted_score:.4f})")
            except Exception as e:
                logging.error(f"Error evaluating {dataset_name}: {str(e)}")
                continue
        
        return total_score

    def train_model(self, model, train_loader, loss_function):
        """Training logic with evolved loss function"""
        optimizer = torch.optim.Adam(model.parameters())
        model.train()
        
        for idx, (inputs, targets) in enumerate(train_loader):
            if idx == self.config.Validator.train_batches:  # Limit training time
                break
                
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if hasattr(model, 'output_size') and model.output_size > 1:
                targets = F.one_hot(targets, num_classes=model.output_size).float()
            
            loss = self.safe_evaluate(loss_function, outputs, targets)
            loss.backward()
            optimizer.step()

    def evaluate_single_dataset(self, model, val_loader):
        """Evaluation logic for a single dataset"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return correct / total if total > 0 else 0.0

    @staticmethod
    def safe_evaluate(func, outputs, labels):
        """Safe evaluation of potentially unstable loss functions"""
        try:
            loss = func(outputs, labels)
            
            if loss is None or not torch.is_tensor(loss):
                return torch.tensor(float('inf'), device=outputs.device)
            
            if not torch.isfinite(loss).all():
                return torch.tensor(float('inf'), device=outputs.device)
            
            if loss.ndim > 0:
                loss = loss.mean()
            
            return loss
        except Exception as e:
            logging.error(f"Error in loss calculation: {str(e)}")
            return torch.tensor(float('inf'), device=outputs.device)

class ValidatorFactory:
    @staticmethod
    def get_validator(config):
        validator_type = config.Validator.validator_type
        if validator_type == "activation":
            return ActivationValidator(config)
        elif validator_type == "loss":
            return LossValidator(config)
        else:
            raise ValueError(f"Unknown validator type: {validator_type}")