import copy
import logging
import math
import threading
import time
from typing import List, Tuple

import bittensor
import bittensor as bt
import bittensor.utils.networking as net
import numpy as np
import torch
from bittensor import logging

from . import __spec_version__


class BittensorNetwork:
    _instance = None
    _lock = threading.Lock()  # Singleton lock
    _weights_lock = threading.Lock()  # Lock for set_weights
    _anomaly_lock = threading.Lock()  # Lock for detect_metric_anomaly
    _config_lock = threading.Lock()  # Lock for modifying config
    _rate_limit_lock = threading.Lock()
    metrics_data = {}
    model_checksums = {}
    request_counts = {}  # Track request counts
    blacklisted_addresses = {}  # Track blacklisted addresses
    last_sync_time = 0
    sync_interval = 600

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(BittensorNetwork, cls).__new__(cls)
                cls.wallet = None
                cls.subtensor = None
                cls.metagraph = None
                cls.config = None
        return cls._instance

    @classmethod
    def initialize(cls, config, ignore_regs=False):
        with cls._lock:
            cls.wallet = bt.wallet(config=config)
            cls.subtensor = bt.subtensor(config=config)
            cls.metagraph = cls.subtensor.metagraph(config.netuid)
            cls.config = config
            if (
                not cls.subtensor.is_hotkey_registered(
                    netuid=config.netuid, hotkey_ss58=cls.wallet.hotkey.ss58_address
                )
                and not ignore_regs
            ):
                logging.error(
                    f"Wallet: {config.wallet} is not registered on netuid {config.netuid}. Please register the hotkey before trying again"
                )
                exit()
                cls.uid = cls.metagraph.hotkeys.index(cls.wallet.hotkey.ss58_address)
            else:
                cls.uid = 0
            cls.device = "cpu"
            cls.base_scores = torch.zeros(
                cls.metagraph.n, dtype=torch.float32, device=cls.device
            )
        # Additional initialization logic here

    @classmethod
    def set_weights(cls, scores):
        try:
            # chain_weights = torch.zeros(cls.subtensor.subnetwork_n(netuid=cls.metagraph.netuid))
            uids = []
            for uid, public_address in enumerate(cls.metagraph.hotkeys):
                try:
                    # alpha = 0.333333 # T=5 (2/(5+1))
                    cls.base_scores[uid] = scores.get(public_address, 0)
                    uids.append(uid)
                except KeyError:
                    continue
            uids = torch.tensor(uids)
            logging.info(f"raw_weights {cls.base_scores}")
            logging.info(f"raw_weight_uids {uids}")
            # Process the raw weights to final_weights via subtensor limitations.
            (
                processed_weight_uids,
                processed_weights,
            ) = bt.utils.weight_utils.process_weights_for_netuid(
                uids=uids.to("cpu").detach().numpy(),
                weights=cls.base_scores.to("cpu").detach().numpy(),
                netuid=cls.config.netuid,
                subtensor=cls.subtensor,
                metagraph=cls.metagraph,
            )

            logging.info(f"processed_weights {processed_weights}")
            logging.info(f"processed_weight_uids {processed_weight_uids}")

            # Convert to uint16 weights and uids.
            (
                uint_uids,
                uint_weights,
            ) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
                uids=processed_weight_uids, weights=processed_weights
            )
            logging.info("Sending weights to subtensor")

            result = cls.subtensor.set_weights(
                wallet=cls.wallet,
                netuid=cls.metagraph.netuid,
                uids=uint_uids,
                weights=uint_weights,
                wait_for_inclusion=False,
                version_key=__spec_version__,
            )
        except Exception as e:
            logging.info(f"Error setting weights: {e}")

    @classmethod
    def get_validator_uids(cls, vpermit_tao_limit: int = 1024):
        """
        Check availability of all UIDs in a given subnet, returning their IP, port numbers, and hotkeys
        if they are serving and have at least vpermit_tao_limit stake, along with a list of strings
        formatted as 'ip:port' for each validator.

        Args:
            metagraph (bt.metagraph.Metagraph): Metagraph object.
            vpermit_tao_limit (int): Validator permit tao limit.

        Returns:
            Tuple[List[dict], List[str]]: A tuple where the first element is a list of dicts with details
                                            of available UIDs, including their IP, port, and hotkeys, and the
                                            second element is a list of strings formatted as 'ip:port'.
        """
        validator_uids = []  # List to hold 'ip:port' strings
        for uid in range(len(cls.metagraph.S)):
            if cls.metagraph.S[uid] >= vpermit_tao_limit:
                validator_uids.append(uid)
        return validator_uids

    @classmethod
    def should_set_weights(cls) -> bool:
        with cls._lock:  # Assuming last_update modification is protected elsewhere with the same lock
            return (
                cls.subtensor.get_current_block() - cls.metagraph.last_update[cls.uid]
            ) > cls.config.epoch_length

    @classmethod
    def resync_metagraph(cls, lite=True):
        # Fetch the latest state of the metagraph from the Bittensor network
        logging.info("Resynchronizing metagraph...")
        # Update the metagraph with the latest information from the network
        cls.metagraph = cls.subtensor.metagraph(cls.config.netuid, lite=lite)
        if not cls.subtensor.is_hotkey_registered(
            netuid=cls.config.netuid, hotkey_ss58=cls.wallet.hotkey.ss58_address
        ):
            logging.error(
                f"Wallet: {cls.config.wallet} is not registered on netuid {cls.config.netuid}. Please register the hotkey before trying again"
            )
            exit()
            cls.uid = cls.metagraph.hotkeys.index(cls.wallet.hotkey.ss58_address)

        logging.info("Metagraph resynchronization complete.")

    @staticmethod
    def should_sync_metagraph(last_sync_time, sync_interval):
        current_time = time.time()
        return (current_time - last_sync_time) > sync_interval

    @classmethod
    def sync(cls, lite=True):
        if cls.should_sync_metagraph(cls.last_sync_time, cls.sync_interval):
            # Assuming resync_metagraph is a method to update the metagraph with the latest state from the network.
            # This method would need to be defined or adapted from the BaseNeuron implementation.
            try:
                cls.resync_metagraph(lite)
                cls.last_sync_time = time.time()
            except Exception as e:
                logging.warning(f"Failed to resync metagraph: {e}")
        else:
            logging.info("Metagraph Sync Interval not yet passed")
