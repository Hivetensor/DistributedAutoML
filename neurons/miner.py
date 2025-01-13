from dml.chain.btt_connector import BittensorNetwork
from dml.configs.config import config
from dml.miners import MinerFactory
import logging


def main(config):
    try:
        bt_config = config.get_bittensor_config()
        BittensorNetwork.initialize(bt_config)
        config.bittensor_network = BittensorNetwork

        miner = MinerFactory.get_miner(config)

        best_genome = miner.mine()

        if hasattr(best_genome, "fitness"):
            print(f"Best genome fitness: {best_genome.fitness.values[0]:.4f}")
            if hasattr(miner, "baseline_accuracy"):
                print(f"Baseline accuracy: {miner.baseline_accuracy:.4f}")
                print(
                    f"Improvement over baseline: {best_genome.fitness.values[0] - miner.baseline_accuracy:.4f}"
                )

        return best_genome

    except Exception as e:
        logging.error(f"Fatal error in main: {str(e)}")
        raise


if __name__ == "__main__":
    best_genome = main(config)
