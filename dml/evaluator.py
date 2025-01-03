
from deap import base, creator, gp, tools

from dml.data import load_datasets
from dml.gene_io import load_individual_from_json
from dml.models import get_model_for_dataset
from dml.ops import create_pset, create_pset_validator, batch_loss

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import logging
import numpy as np
import os
from typing import List, Dict, Any

import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch 
from tqdm import tqdm 
from typing import Callable, Dict, List


class LossEvaluator:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def safe_evaluate(func: Callable, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Safely evaluate a loss function."""
        try:
            loss = func(outputs, labels)
            if loss is None or not torch.is_tensor(loss) or not torch.isfinite(loss).all():
                return torch.tensor(float('inf'), device=outputs.device)
            return loss.mean() if loss.ndim > 0 else loss
        except Exception as e:
            logging.error(f"Error in loss calculation: {str(e)}")
            return torch.tensor(float('inf'), device=outputs.device)

    def train_and_evaluate(
        self,
        model: torch.nn.Module,
        loss_function: Callable,
        train_loader: DataLoader,
        val_loader: DataLoader, 
        num_classes: int = 10,
        metric_type: str = "loss",
        learning_rate: float = 0.001,
    ) -> Dict[str, List[float]]:
        """Train the model and evaluate its performance."""
        metrics = {
            'train_loss': [],
            'val_accuracy': [],
            'batch_numbers': []
        }
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        batch_counter = 0
        
        model.train()
        for epoch in range(self.config.Evaluator.epochs):
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}") as pbar:
                for inputs, targets in train_loader:
                    batch_counter += 1
                    if batch_counter > self.config.Evaluator.max_batches:
                        break
                    
                    metrics = self._training_step(
                        model, loss_function, optimizer,
                        inputs, targets, metrics, batch_counter, num_classes
                    )
                    
                    if batch_counter % self.config.Evaluator.validate_every == 0:
                        metrics = self._validation_step(
                            model, val_loader, metrics, batch_counter, metric_type
                        )
                        pbar.set_postfix({
                            'Loss': f"{metrics['train_loss'][-1]:.4f}",
                            'Val Acc': f"{metrics['val_accuracy'][-1]:.4f}"
                        })
                    
                    pbar.update(1)
        
        return metrics
    
    def _training_step(
        self,
        model: torch.nn.Module,
        loss_function: Callable,
        optimizer: torch.optim.Optimizer,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        metrics: Dict[str, List[float]],
        batch_counter: int,
        num_classes: int = 10
    ) -> Dict[str, List[float]]:
        """Perform a single training step."""
        inputs = inputs.to(self.config.device)
        targets = targets.to(self.config.device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        print(num_classes)
        print(max(targets))
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
        try:
            loss = self.safe_evaluate(loss_function, outputs, targets_one_hot)
            loss.backward()
            optimizer.step()
            
            metrics['train_loss'].append(loss.item())
        except:
            pass
            #breakpoint()
        
        return metrics

    def _validation_step(
        self,
        model: torch.nn.Module,
        val_loader: DataLoader,
        metrics: Dict[str, List[float]],
        batch_counter: int,
        metric_type: str = 'loss'  # New parameter to choose metric type
    ) -> Dict[str, List[float]]:
        """Perform validation and update metrics."""
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for idx, (val_inputs, val_targets) in tqdm(enumerate(val_loader)):
                val_inputs = val_inputs.to(self.config.device)
                val_targets = val_targets.to(self.config.device)
                val_outputs = model(val_inputs)
                if len(val_outputs.shape) == 3:
                    if idx > self.config.Evaluator.llm_validation_steps: #max validation config
                        break
                    _, predicted = val_outputs.max(dim=-1)
                    total += val_targets.numel()  # Count all elements
                    correct += predicted.eq(val_targets).sum().item()

                    loss = F.cross_entropy(
                    val_outputs.view(-1, val_outputs.size(-1)), 
                    val_targets.view(-1)
                    )
                    total_loss += loss.item()
                else:
                    _, predicted = val_outputs.max(1)
                    total += val_targets.size(0)
                    correct += predicted.eq(val_targets).sum().item()

                    loss = F.cross_entropy(val_outputs, val_targets)
                    total_loss += loss.item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(val_loader)
        if metric_type == 'accuracy':
            metrics['val_accuracy'].append(accuracy)
        else:
            metrics['val_accuracy'].append(avg_loss) #FIXME Hack 
        metrics['batch_numbers'].append(batch_counter)
        model.train()
        return metrics
    
class ComplexityLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

@dataclass
class LossFunctionResult:
    name: str
    mnist_accuracy: float
    training_time: float
    complexity: ComplexityLevel
    accuracy_progression: List[float]
    cifar10_accuracy: float = None  # Optional for now


class ResultsHandler:
    def __init__(self):
        self.results: List[LossFunctionResult] = []
    
    def add_result(self, result: LossFunctionResult):
        """Add a new result to the collection."""
        self.results.append(result)

    def _determine_complexity(self, function_str: str) -> ComplexityLevel:
        """Determine complexity based on function structure."""
        # Simple heuristic based on function length and number of operations
        if len(function_str) < 50:
            return ComplexityLevel.LOW
        elif len(function_str) < 150:
            return ComplexityLevel.MEDIUM
        return ComplexityLevel.HIGH

    def process_evaluation_metrics(self, 
                                 name: str,
                                 metrics: Dict[str, List[float]],
                                 function_str: str = None,
                                 total_batches: int = None,
                                 epochs: int = None) -> LossFunctionResult:
        """Process raw evaluation metrics into a structured result."""
        # Get the final accuracy and create progression list
        accuracy_values = metrics['val_accuracy']
        final_accuracy = accuracy_values[-1] * 100  # Convert to percentage
        
        # Calculate training time as proportion of total possible iterations
        if total_batches and epochs:
            training_time = len(metrics['train_loss']) / (total_batches * epochs)
        else:
            training_time = 1.0  # Default to 1.0 if not specified
        
        # Determine complexity
        complexity = (self._determine_complexity(function_str) 
                     if function_str else ComplexityLevel.LOW)
        
        # Create result object
        result = LossFunctionResult(
            name=name,
            mnist_accuracy=final_accuracy,
            training_time=training_time,
            complexity=complexity,
            accuracy_progression=[acc * 100 for acc in accuracy_values]  # Convert to percentages
        )
        
        self.add_result(result)
        return result

    def generate_json_output(self, output_file: str = None) -> Dict[str, Any]:
        """Generate JSON output in the specified format."""
        output = {
            "lastUpdated": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sota": [
                {
                    "name": result.name,
                    "mnist_accuracy": round(result.mnist_accuracy, 1),
                    "cifar10_accuracy": round(result.cifar10_accuracy, 1) if result.cifar10_accuracy else None,
                    "training_time": round(result.training_time, 2),
                    "complexity": result.complexity.value
                }
                for result in self.results
            ],
            "performance": [
                {
                    "name": result.name,
                    "values": [round(v, 2) for v in result.accuracy_progression]
                }
                for result in self.results
            ]
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(output, f, indent=4)
        
        return output

class TaskEvaluator:
    def __init__(self, config):
        self.config = config
        self.evaluator = LossEvaluator(config)
        self.results_handler = ResultsHandler()  # Add this line
        self.initialize_deap()

    def initialize_deap(self):
        self.toolbox = base.Toolbox()
        self.pset = create_pset_validator()

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=3)
        self.toolbox.register(
            "individual", tools.initIterate, creator.Individual, self.toolbox.expr
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        #self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register(
            "mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset
        )

    def evaluate_loss_functions(self, json_folder: str) -> None:
        """Evaluate multiple loss functions from JSON files."""
        for dataset in self.config.Evaluator.architectures.keys():
            dataset_spec = load_datasets(dataset)[0]
            train_loader = dataset_spec.train_loader
            val_loader = dataset_spec.val_loader
            for architecture in self.config.Evaluator.architectures[dataset]:

                self.results_handler = ResultsHandler()  

                for filename in tqdm(os.listdir(json_folder)):
                    if filename.endswith('.json'):
                        self._evaluate_single_loss(
                            os.path.join(json_folder, filename),
                            train_loader,
                            val_loader,
                            dataset, 
                            architecture
                        )

                self._evaluate_baseline_losses(train_loader, val_loader, dataset,architecture)
            
                # Update visualization call:
                

                
                # Add JSON output generation:
                output_file = f"evaluation_results_{dataset}_{architecture}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.results_handler.generate_json_output(output_file)

    def _evaluate_single_loss(
        self,
        file_path: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        dataset: str,
        arch: str
    ) -> Dict[str, Any]:
        """Evaluate a single loss function from a JSON file."""
        #try
        torch.manual_seed(self.config.seed)
        individual, loss_function, loss_str, _ = load_individual_from_json(
            filename=file_path,
            pset=self.pset,
            toolbox=self.toolbox
        )
        
        model = get_model_for_dataset(dataset, arch)
        model.to(self.config.device)
        if dataset == "shakespeare":
            metrics = self.evaluator.train_and_evaluate(
                model, loss_function, train_loader, val_loader, num_classes=85, metric_type="loss"
            )
        elif dataset == "fineweb":
            metrics = self.evaluator.train_and_evaluate(
                model, loss_function, train_loader, val_loader, num_classes=50257, metric_type="loss"
            )
        elif (dataset == "cifar100") or (dataset == "fgvc_aircraft"):
            metrics = self.evaluator.train_and_evaluate(
                model, loss_function, train_loader, val_loader, num_classes=100, metric_type="accuracy"
            )            
        elif dataset == "food101": #FIXME This is an ugly hack fix it and load num_classes from the dataset spec 
            metrics = self.evaluator.train_and_evaluate(
                model, loss_function, train_loader, val_loader, num_classes=101, metric_type="accuracy"
            )            
        else:
            metrics = self.evaluator.train_and_evaluate(
                model, loss_function, train_loader, val_loader, metric_type="accuracy"
            )
        
        # Add results processing:
        self.results_handler.process_evaluation_metrics(
            name=os.path.basename(file_path),
            metrics=metrics,
            function_str=str(individual),
            total_batches=len(train_loader),
            epochs=self.config.Evaluator.epochs
        )
        # except Exception as e:
        #     print(e)
        #     metrics = {
        #         'train_loss': [99],
        #         'val_accuracy': [0.0],
        #         'batch_numbers': [1]
        #     }
        #     self.results_handler.process_evaluation_metrics(
        #         name=os.path.basename(file_path),
        #         metrics=metrics,
        #         function_str=None,
        #         total_batches=len(train_loader),
        #         epochs=self.config.epochs
        #     )


    def _evaluate_baseline_losses(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        dataset: str,
        arch: str
    ) -> List[Dict[str, Any]]:
        """Evaluate baseline loss functions (MSE and Cross-Entropy)."""
        baseline_losses = {
            'MSE': torch.nn.MSELoss(),
            'CrossEntropy': torch.nn.CrossEntropyLoss(),
            "evolved": batch_loss
        }

        for loss_name, loss_fn in baseline_losses.items():
            torch.manual_seed(self.config.seed)
            model = get_model_for_dataset(dataset, arch)
            model.to(self.config.device)
            
            #try:
            if loss_name == 'MSE':
                loss_function = lambda outputs, targets: loss_fn(outputs, targets)
            else:  # CrossEntropy
                loss_function = torch.nn.CrossEntropyLoss()

            if dataset == "shakespeare":
                metrics = self.evaluator.train_and_evaluate(
                    model, loss_function, train_loader, val_loader, num_classes=85, metric_type="loss"
                )
            elif dataset == "fineweb":
                metrics = self.evaluator.train_and_evaluate(
                    model, loss_function, train_loader, val_loader, num_classes=50257, metric_type="loss"
                )
            elif (dataset == "cifar100") or (dataset == "fgvc_aircraft"):
                metrics = self.evaluator.train_and_evaluate(
                    model, loss_function, train_loader, val_loader, num_classes=100, metric_type="accuracy"
                )
            elif dataset == "food101": #FIXME This is an ugly hack fix it and load num_classes from the dataset spec 
                metrics = self.evaluator.train_and_evaluate(
                    model, loss_function, train_loader, val_loader, num_classes=101, metric_type="accuracy"
                )
            else:
                metrics = self.evaluator.train_and_evaluate(
                    model, loss_function, train_loader, val_loader, metric_type="accuracy"
                )

            
            # Add results processing:
            self.results_handler.process_evaluation_metrics(
                name=f"{loss_name} (Baseline)",
                metrics=metrics,
                total_batches=len(train_loader),
                epochs=self.config.Evaluator.epochs
            )
            # except:
            #     continue

