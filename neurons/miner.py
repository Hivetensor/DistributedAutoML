from dml.chain.btt_connector import BittensorNetwork
from dml.configs.config import config
from dml.miners import MinerFactory
import logging


def main(config):
    try:
        bt_config = config.get_bittensor_config()
        BittensorNetwork.initialize(bt_config)
        config.bittensor_network = BittensorNetwork

        if config.Miner.push_platform == "pool":
            if not hasattr(config.Miner, 'pool_url'):
                raise ValueError("pool_url must be specified in config when using pool platform")
            if not hasattr(config.Miner, 'miner_operation'):
                raise ValueError("miner_operation must be specified when using pool platform")

        miner = MinerFactory.get_miner(config)

        if config.Miner.push_platform == "pool":
            logging.info(f"Starting pool {config.Miner.miner_operation} operation as {config.Miner.miner_type} miner")
        else:
            logging.info(f"Starting standalone {config.Miner.miner_type} miner")

        best_genome = miner.mine()

        # Print results if evolution happened
        if hasattr(best_genome, 'fitness'):
            print(f"Best genome fitness: {best_genome.fitness.values[0]:.4f}")
            if hasattr(miner, 'baseline_accuracy'):
                print(f"Baseline accuracy: {miner.baseline_accuracy:.4f}")
                print(f"Improvement over baseline: {best_genome.fitness.values[0] - miner.baseline_accuracy:.4f}")

        return best_genome

    except Exception as e:
        logging.error(f"Fatal error in main: {str(e)}")
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    config.Miner.miner_type = "loss"
    config.Miner.push_platform = "hf"  # Options: "pool", "hf" (Hugging Face)

    # Pool-specific config only needed if using pool
    if config.Miner.push_platform == "pool":
        config.Miner.miner_operation = "evolve"  # or "evaluate"

    # Run main
    best_genome = main(config)
