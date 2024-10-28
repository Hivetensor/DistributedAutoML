import argparse
import json
import multiprocessing
import subprocess

import bittensor as bt
from validator import main as validator_main

from dml.configs.config import config
from miner import main as miner_main


def run_node(node_type, wallet_info, config):
    if node_type == "miner":
        config.Bittensor.wallet_name = wallet_info["coldkey_name"]
        config.Bittensor.wallet_hotkey = wallet_info["hotkey_name"]
        config.gene_repo = "mekaneeky/" + wallet_info["hf_repo"]
        miner_main(config)

    elif node_type == "validator":
        config.Bittensor.wallet_name = wallet_info["coldkey_name"]
        config.Bittensor.wallet_hotkey = wallet_info["hotkey_name"]
        config.gene_repo = "mekaneeky/" + wallet_info["hf_repo"]
        validator_main(config)

    else:
        raise ValueError(f"Invalid node type: {node_type}")


def load_wallet_info(file_path):
    with open(file_path, "r") as file:
        wallet_data = json.load(file)
    return wallet_data


def main():
    parser = argparse.ArgumentParser(
        description="Script to launch multiple miners and validators."
    )
    parser.add_argument(
        "--wallet_info_file", required=True, help="Path to the wallet info JSON file"
    )
    parser.add_argument(
        "--n_miners", type=int, default=5, help="Number of miners to launch"
    )
    parser.add_argument(
        "--n_validators", type=int, default=1, help="Number of validators to launch"
    )

    args = parser.parse_args()
    miner_count = 0
    validator_count = 0
    # Load wallet info
    wallet_info_list = load_wallet_info(args.wallet_info_file)

    processes = []

    if args.n_miners > 0:
        miner_wallet_info = wallet_info_list[: args.n_miners]
        for miner_wallet in miner_wallet_info:
            config_node = config()
            config_node.metrics_file = f"miner_{miner_count}.csv"
            miner_count += 1
            p = multiprocessing.Process(
                target=run_node, args=("miner", miner_wallet, config_node)
            )
            p.start()
            processes.append(p)

    if args.n_validators > 0:
        validator_wallet_info = wallet_info_list[89 : 89 + args.n_validators]
        for validator_wallet in validator_wallet_info:
            config_node = config()
            config_node.metrics_file = f"validator_{validator_count}.csv"
            validator_count += 1
            p = multiprocessing.Process(
                target=run_node, args=("validator", validator_wallet, config_node)
            )
            p.start()
            processes.append(p)


if __name__ == "__main__":
    main()
