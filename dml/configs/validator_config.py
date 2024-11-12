import numpy as np

def constrained_decay(n: int, ratio: float = 5.0):
    smallest = 2 / (n * (1 + ratio))
    largest = smallest * ratio
    decay = np.linspace(largest, smallest, n)
    normalized_decay = decay / np.sum(decay)
   
    return normalized_decay.tolist() 

class ValidatorConfig:
    dataset_names = ["cifar10", "shakespeare"]
    validation_interval = 2000  # Interval between validations in seconds
    validator_type = "loss"
    top_k = 50  # Number of top miners to distribute scores to
    min_score = 0.0  # Minimum score for miners not in the top-k
    top_k_weight = constrained_decay(50, 5.0)
    time_penalty_factor = 0.5
    time_penalty_max_time = 7200  #1week
    max_gene_size = 1024*20
    seed = 42
    training_iterations = 100
    validation_iterations = 100

    
