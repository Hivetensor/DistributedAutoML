import numpy as np
import torch
import random

def tensor_to_numpy(tensor):
    return tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()

def numpy_to_tensor(array):
    return torch.from_numpy(array)

def ensure_numpy(data):
    if isinstance(data, torch.Tensor):
        return tensor_to_numpy(data)
    elif isinstance(data, np.ndarray):
        return data
    else:
        return np.array(data)

def ensure_tensor(data):
    if isinstance(data, np.ndarray):
        return numpy_to_tensor(data)
    elif isinstance(data, torch.Tensor):
        return data
    else:
        return torch.tensor(data)
    
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

import re

def calculate_tree_depth(expression: str) -> int:
    """Calculates the maximum depth of nested function calls in an expression.

    Args:
        expression: The expression string to analyze.

    Returns:
        The maximum depth of nested function calls.
    """
    # Tokenize the expression to detect function calls.
    tokens = re.findall(r'\w+\(|\)|\w+', expression)
    max_depth = 0
    current_depth = 0

    for token in tokens:
        if token.endswith('('):  # A function call starts
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif token == ')':  # A function call ends
            current_depth -= 1

    if current_depth != 0:
        raise ValueError("Mismatched parentheses in the expression.")

    return max_depth


def get_validator_urls(metagraph) -> list[str]:
    """Get validator URLs from metagraph."""
    validator_urls = []
    
    for uid in range(len(metagraph.uids)):
        # Check if validator has permit and sufficient stake
        if metagraph.S[uid] > 5:
            # Get validator URL from axon info
            if metagraph.axons[uid].ip and metagraph.axons[uid].port:
                if metagraph.axons[uid].ip == "55.55.55.55":
                    #for testing purposes
                    url = f"http://127.0.0.1:8000"
                else:
                    url = f"http://{metagraph.axons[uid].ip}:{metagraph.axons[uid].port}"
                validator_urls.append(url)
    
    return validator_urls