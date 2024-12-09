import unittest
from dml.ops import create_pset_validator
from dml.gene_io import safe_eval
from dml.gp_fix import SafePrimitiveTree
from dml.deap_individual import Individual
from deap import creator, gp, base, tools
from dml.utils import set_seed
from dml.configs.config import config
from dml.data import load_datasets
from dml.validators import ValidatorFactory
from dml.chain.btt_connector import BittensorNetwork
from dml.chain.chain_manager import ChainManager
import numpy as np

class TestDeterminism(unittest.TestCase):
    def test_evaluation(self, seed=42):
        """
        Ensures that scoring is consistent across validators
        """
        # Initialize DEAP
        pset = create_pset_validator()
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        # Initialize validators
        bt_config = config.get_bittensor_config()
        BittensorNetwork.initialize(bt_config)
        config.bittensor_network = BittensorNetwork
        config.chain_manager = ChainManager(
            subtensor=BittensorNetwork.subtensor,
            subnet_uid=bt_config.netuid,
            wallet=BittensorNetwork.wallet,
        )
        validator1 = ValidatorFactory.get_validator(config)
        validator2 = ValidatorFactory.get_validator(config)

        # Load data
        set_seed(seed)
        datasets = load_datasets(config.Validator.dataset_names, batch_size=32, seed=seed)

        # Compile and evaluate test genes
        loss1 = 'square(safe_sub(x, y))'
        loss2 = 'cube(safe_sub(x, y))'
        gene1 = Individual(SafePrimitiveTree.from_string(loss1, pset, safe_eval))
        gene2 = Individual(SafePrimitiveTree.from_string(loss2, pset, safe_eval))

        # Validator 1's scores
        val1_fitness1 = validator1.evaluate_individual(gene1, datasets).cpu().detach().numpy()
        val1_fitness2 = validator1.evaluate_individual(gene2, datasets).cpu().detach().numpy()

        # Validator 2's scores
        val2_fitness2 = validator2.evaluate_individual(gene2, datasets).cpu().detach().numpy()
        val2_fitness1 = validator2.evaluate_individual(gene1, datasets).cpu().detach().numpy()

        self.assertTrue(np.allclose(val1_fitness1, val2_fitness1))
        self.assertTrue(np.allclose(val1_fitness2, val2_fitness2))

if __name__ == '__main__':
    unittest.main()
