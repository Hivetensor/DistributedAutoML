import json
import random
import unittest

from deap import gp
from expression_transformer import transform_expression

from dml.ops import (
    safe_add,
    safe_div,
    safe_mul,
    safe_relu,
    safe_sigmoid,
    safe_sub,
    safe_tanh,
)

# Assuming these are defined elsewhere
from dml.serialize import ALLOWED_PRIMITIVES, serialize_primitive_tree


def generate_random_expression(depth=0, max_depth=3):
    if depth == max_depth or random.random() < 0.3:
        return random.choice(["x", "y", "z", "const_1.0", "const_2.0", "const_0.5"])

    ops = ["safe_add", "safe_sub", "safe_mul", "safe_div"]
    op = random.choice(ops)
    left = generate_random_expression(depth + 1, max_depth)
    right = generate_random_expression(depth + 1, max_depth)
    return f"{op}({left}, {right})"


def execute_genome(genome, inputs):
    # This is a placeholder function
    # In a real scenario, this would execute the genome and return a result
    return 0


import math
import operator
import random
import unittest

from deap import base, creator, gp, tools

# Define the primitive set
pset = gp.PrimitiveSet("MAIN", 3)  # 3 inputs: x, y, z
pset.addPrimitive(operator.add, 2, name="safe_add")
pset.addPrimitive(operator.sub, 2, name="safe_sub")
pset.addPrimitive(operator.mul, 2, name="safe_mul")
pset.addPrimitive(lambda x, y: x / y if y != 0 else 1, 2, name="safe_div")
pset.addTerminal(1.0, name="const_1.0")
pset.addTerminal(2.0, name="const_2.0")
pset.addTerminal(0.5, name="const_0.5")
pset.renameArguments(ARG0="x", ARG1="y", ARG2="z")

# Create fitness and individual classes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# Create the toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def execute_genome(genome, inputs):
    # This is a placeholder function for executing transformed genomes
    # In a real scenario, this would execute the transformed genome and return a result

    return 0


class TestGenomeTransformer(unittest.TestCase):
    def test_genome_transformation(self):
        num_tests = 1000
        for _ in range(num_tests):
            # Generate a random genome using DEAP
            individual = toolbox.individual()
            original_genome = str(individual)

            # Transform the genome
            transformed_genome = transform_expression(original_genome)

            # Generate random inputs
            inputs = {
                "x": random.uniform(-1, 1),
                "y": random.uniform(-1, 1),
                "z": random.uniform(-1, 1),
            }

            # Execute original genome using DEAP
            func = gp.compile(individual, pset)
            original_result = func(inputs["x"], inputs["y"], inputs["z"])

            # Execute transformed genome (placeholder)
            transformed_result = execute_genome(transformed_genome, inputs)

            # Compare results
            self.assertAlmostEqual(
                original_result,
                transformed_result,
                places=5,
                msg=f"Mismatch for genome: {original_genome}",
            )


if __name__ == "__main__":
    unittest.main()
