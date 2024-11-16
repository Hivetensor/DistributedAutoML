
from dml.chain.btt_connector import BittensorNetwork
from dml.chain.chain_manager import ChainMultiAddressStore
from dml.configs.config import config

import logging

from bittensor.core.extrinsics.serving import serve_extrinsic

def main(config):
    bt_config = config.get_bittensor_config()
    BittensorNetwork.initialize(bt_config)

    
    served = serve_extrinsic(
        subtensor=BittensorNetwork.subtensor,
        wallet=BittensorNetwork.wallet,
        ip=config.Validator.external_ip,
        port=config.Validator.external_port,
        netuid=config.Bittensor.netuid,
        protocol=4,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    
    if served:
        logging.info("Serving request posted succesfully. Confirm address from metagraph")
    else:
        logging.error("Serving request failed to post")
    

if __name__ == "__main__":
    main(config)
