from dml.chain.btt_connector import BittensorNetwork
from dml.chain.chain_manager import ChainMultiAddressStore
from dml.configs.config import config


def main(config):
    bt_config = config.get_bittensor_config()
    BittensorNetwork.initialize(bt_config)

    # Initialize Chain Manager and HF Manager
    chain_manager = ChainMultiAddressStore(
        BittensorNetwork.subtensor, bt_config.netuid, BittensorNetwork.wallet
    )

    print(f"Storing repo {config.gene_repo} to chain")
    chain_manager.store_hf_repo(config.gene_repo)

    print("Ensuring successful push")
    retrieved_repo = chain_manager.retrieve_hf_repo(
        BittensorNetwork.wallet.hotkey.ss58_address
    )

    if retrieved_repo != config.gene_repo:
        raise ValueError(
            f"Retrieved repo value:'{retrieved_repo}' not equal to config repo '{config.gene_repo}'"
        )
    else:
        print(f"Correct repo '{retrieved_repo}' retrived ! Storing successful.")


if __name__ == "__main__":
    main(config)
