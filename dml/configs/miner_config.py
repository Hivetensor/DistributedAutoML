import time 
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class DatasetConfig:
    """Configuration for a single dataset"""
    enabled: bool = True
    weight: float = 0.25
    batch_size: int = 32
    training_iterations: int = 100
    validation_iterations: int = 10
    sequence_length: Optional[int] = None  # For text datasets
    num_samples: Optional[int] = None      # For subset sampling

class MinerConfig:
     # Dataset-specific configurations
    datasets = {
        'mnist': DatasetConfig(
            enabled=True,
            weight=0.2,
            batch_size=64,
            training_iterations=100,
            validation_iterations=10
        ),
        'cifar10': DatasetConfig(
            enabled=True,
            weight=0.3,
            batch_size=32,
            training_iterations=150,
            validation_iterations=15
        ),
        'shakespeare': DatasetConfig(
            enabled=True,
            weight=0.3,
            batch_size=16,
            training_iterations=50,
            validation_iterations=5,
            sequence_length=128
        ),
        'imagenet_subset': DatasetConfig(
            enabled=True,
            weight=0.2,
            batch_size=16,
            training_iterations=75,
            validation_iterations=8,
            num_samples=1000
        )
    }
        
        # Training parameters
    training_params = {
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'gradient_clip': 1.0
        }
    #TODO limit validator memory allowed to prevent DOS attacks 
    checkpoint_save_dir = "checkpoints"
    check_registration_interval = 500
    evaluation_iterations = 10
    gp_tree_height = 90
    generations = 100000
    migration_interval = 100
    migrants_per_round = 10
    miner_type = "loss"
    num_processes = 1
    pool_url = None #"http://127.0.0.1:5000"
    population_size = 200 # Per process pop = population_size // num_processes
    push_platform = "hf"
    save_temp_only = True
    seed = int(time.time())
    tournament_size = 2    
    training_iterations = 10
    
