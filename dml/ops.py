import operator
import random

import torch
from deap import gp

from dml.configs.config import config

device = torch.device(config.device)


def safe_div(x, y):
    epsilon = 1e-8
    return x / (y + epsilon)


def safe_add(x, y):
    x = torch.tensor(x, device=device) if not torch.is_tensor(x) else x
    y = torch.tensor(y, device=device) if not torch.is_tensor(y) else y
    return x + y


def safe_sub(x, y):
    x = torch.tensor(x, device=device) if not torch.is_tensor(x) else x
    y = torch.tensor(y, device=device) if not torch.is_tensor(y) else y
    return x - y


def safe_mul(x, y):
    x = torch.tensor(x, device=device) if not torch.is_tensor(x) else x
    y = torch.tensor(y, device=device) if not torch.is_tensor(y) else y
    return x * y


def safe_sigmoid(x):
    x = torch.tensor(x, device=device) if not torch.is_tensor(x) else x
    return torch.sigmoid(x)


def safe_relu(x):
    x = torch.tensor(x, device=device) if not torch.is_tensor(x) else x
    return torch.relu(x)


def safe_tanh(x):
    x = torch.tensor(x, device=device) if not torch.is_tensor(x) else x
    return torch.tanh(x)


def safe_log(x):
    return torch.log(torch.abs(x) + 1e-8)


def safe_sqrt(x):
    return torch.sqrt(torch.abs(x))


def safe_exp(x):
    return torch.exp(torch.clamp(x, -100, 100))


def generate_random():
    return torch.tensor(random.uniform(-1, 1), device=device)


def create_pset():
    pset = gp.PrimitiveSet("MAIN", 2)

    # Basic arithmetic operations
    pset.addPrimitive(safe_add, 2)
    pset.addPrimitive(safe_sub, 2)
    pset.addPrimitive(safe_mul, 2)
    pset.addPrimitive(safe_div, 2)

    # Advanced mathematical operations
    pset.addPrimitive(safe_log, 1)
    pset.addPrimitive(safe_sqrt, 1)
    pset.addPrimitive(safe_exp, 1)

    # Trigonometric functions
    pset.addPrimitive(torch.sin, 1)
    pset.addPrimitive(torch.cos, 1)
    pset.addPrimitive(torch.tan, 1)

    # Activation functions
    pset.addPrimitive(torch.sigmoid, 1)
    pset.addPrimitive(torch.relu, 1)
    pset.addPrimitive(torch.tanh, 1)

    # Statistical functions
    pset.addPrimitive(torch.mean, 1)
    pset.addPrimitive(torch.std, 1)

    # Constants

    pset.addEphemeralConstant("rand_const", generate_random)
    pset.addTerminal(torch.tensor(1.0, device=device), name="one")
    pset.addTerminal(torch.tensor(0.0, device=device), name="zero")
    pset.addTerminal(torch.tensor(0.5, device=device), name="half")

    # Rename arguments
    pset.renameArguments(ARG0="x")
    pset.renameArguments(ARG1="y")

    return pset


def create_pset_validator():
    pset = gp.PrimitiveSet("MAIN", 2)

    # Basic arithmetic operations
    pset.addPrimitive(safe_add, 2)
    pset.addPrimitive(safe_sub, 2)
    pset.addPrimitive(safe_mul, 2)
    pset.addPrimitive(safe_div, 2)

    # Advanced mathematical operations
    pset.addPrimitive(safe_log, 1)
    pset.addPrimitive(safe_sqrt, 1)
    pset.addPrimitive(safe_exp, 1)

    # Trigonometric functions
    pset.addPrimitive(torch.sin, 1)
    pset.addPrimitive(torch.cos, 1)
    pset.addPrimitive(torch.tan, 1)
    pset.addPrimitive(torch.arcsin, 1)
    pset.addPrimitive(torch.arccos, 1)
    pset.addPrimitive(torch.arctan, 1)

    # Activation functions
    pset.addPrimitive(torch.sigmoid, 1)
    pset.addPrimitive(torch.relu, 1)
    pset.addPrimitive(torch.tanh, 1)
    pset.addPrimitive(
        lambda x: torch.where(x > 0, x, x * 0.01), 1, name="leaky_relu"
    )  # Leaky ReLU

    # Statistical functions
    pset.addPrimitive(torch.mean, 1)
    pset.addPrimitive(torch.std, 1)

    # Additional operations from FunctionDecoder
    pset.addPrimitive(torch.abs, 1)
    pset.addPrimitive(torch.reciprocal, 1)
    pset.addPrimitive(lambda x: torch.pow(x, 2), 1, name="square")
    pset.addPrimitive(lambda x: torch.pow(x, 3), 1, name="cube")
    pset.addPrimitive(torch.sign, 1)
    pset.addPrimitive(torch.floor, 1)
    pset.addPrimitive(torch.ceil, 1)
    pset.addPrimitive(torch.round, 1)
    pset.addPrimitive(lambda x, y: torch.maximum(x, y), 2, name="max")
    pset.addPrimitive(lambda x, y: torch.minimum(x, y), 2, name="min")
    pset.addPrimitive(lambda x, y: torch.remainder(x, y), 2, name="mod")
    pset.addPrimitive(lambda x, y: torch.hypot(x, y), 2, name="hypot")

    # Constants
    device = torch.device("cpu")
    pset.addEphemeralConstant(
        "rand_const", lambda: torch.tensor(random.uniform(-1, 1), device=device)
    )

    for i in range(1, 9):
        pset.addEphemeralConstant(
            f"rand_const_{i}",
            lambda: torch.tensor(random.uniform(-1, 1), device=device),
        )

    pset.addTerminal(torch.tensor(1.0, device=device), name="one")
    pset.addTerminal(torch.tensor(0.0, device=device), name="zero")
    pset.addTerminal(torch.tensor(0.5, device=device), name="half")
    pset.addTerminal(torch.tensor(3.14159, device=device), name="pi")
    pset.addTerminal(torch.tensor(2.71828, device=device), name="e")

    # Rename arguments
    pset.renameArguments(ARG0="x")
    pset.renameArguments(ARG1="y")

    return pset
