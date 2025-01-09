import logging
import os
import json
from typing import Dict

import requests
import time
import tempfile
from abc import ABC, abstractmethod

from dml.configs.config import config
from dml.gene_io import save_individual_to_json
from dml.chain.chain_manager import SolutionId
from dml.utils import compute_chain_hash

from huggingface_hub import HfApi


class PushDestination(ABC):
    @abstractmethod
    def push(self, gene, commit_message):
        pass


class PushMixin:
    def push_to_remote(self, gene, commit_message):
        if not hasattr(self, 'push_destinations'):
            logging.warning("No push destinations defined. Skipping push to remote.")
            return

        for destination in self.push_destinations:
            destination.push(gene, commit_message)


class HuggingFacePushDestination(PushDestination):
    def __init__(self, repo_name):
        self.repo_name = repo_name
        self.api = HfApi(token=config.hf_token)

    def push(self, gene, commit_message, config, save_temp = config.Miner.save_temp_only):

        if not self.repo_name:
            logging.info("No Hugging Face repository name provided. Skipping push to Hugging Face.")
            return

        # Create a temporary file to store the gene data
        if save_temp:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
                json.dump(save_individual_to_json(gene, hotkey=config.bittensor_network.wallet.hotkey.ss58_address), temp_file)
                temp_file_path = temp_file.name

        else:
            os.makedirs(config.Miner.checkpoint_save_dir, exist_ok=True)

            temp_file_path = os.path.join(
                config.Miner.checkpoint_save_dir,
                f"{commit_message.replace('.', '_')}.json"
            )
            with open(temp_file_path, 'w') as temp_file:
                json.dump(save_individual_to_json(gene, hotkey=config.bittensor_network.wallet.hotkey.ss58_address), temp_file)

        try:
            # if not os.path.exists(self.repo_name):
            #     Repository(self.repo_name, token=config.hf_token ,clone_from=f"https://huggingface.co/{self.repo_name}")

            # repo = Repository(self.repo_name, f"https://huggingface.co/{self.repo_name}",token=config.hf_token)
            # repo.git_pull()

            self.api.upload_file(path_or_fileobj=temp_file_path, path_in_repo=f"best_gene.json", repo_id=self.repo_name,
                                 commit_message=commit_message)
            logging.info(f"Successfully pushed gene to Hugging Face: {commit_message}")
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)


class PoolPushDestination(PushDestination):
    def __init__(self, pool_url, wallet, miner_operation):
        self.pool_url = pool_url
        self.wallet = wallet
        self.miner_operation = miner_operation
        self.register_with_pool()

    def register_with_pool(self) -> bool:
        """Register miner with the pool"""
        try:
            response = self.make_authenticated_request(
                endpoint="miners/register",
                method="post",
                params={},
                data={"public_address": self.wallet.hotkey.ss58_address}
            )

            if not response or response.status_code != 200:
                logging.error(f"Failed to register with pool: {response.text if response else 'No response'}")
                return False

            logging.info(f"Pool registration: {response.json().get('message', 'Success')}")
            return True

        except Exception as e:
            logging.error(f"Critical error during pool registration: {str(e)}")
            return False

    def _prepare_request_data(self, endpoint: str, **data):
        message_dict = {
            "endpoint": endpoint,
            "timestamp": time.time(),
            "data": data
        }
        message = json.dumps(message_dict, sort_keys=True)  # Deterministic serialization

        return {
            "public_address": self.wallet.hotkey.ss58_address,
            "signature": self.wallet.hotkey.sign(message.encode("utf-8")).hex(),
            "message": message,
            "data": data
        }

    def make_authenticated_request(self, endpoint: str,params: Dict, method: str = "post", timeout: int = 30,  **data):
        try:
            request_data = self._prepare_request_data(endpoint, **data)

            if method.lower() == "get":
                response = requests.get(
                    f"{self.pool_url}/{endpoint}",
                    json=request_data,
                    params=params,
                    timeout=timeout
                )
            else:
                response = requests.post(
                    f"{self.pool_url}/{endpoint}",
                    json=request_data,
                    timeout=timeout
                )

            if response.status_code >= 500:
                logging.error(f"Server error: {response.text}")
                return None

            return response

        except requests.exceptions.Timeout:
            logging.error(f"Request to {endpoint} timed out after {timeout} seconds")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {str(e)}")
            return None

    def request_task(self, task_type: str):
        """Get a task from the pool"""
        response = self.make_authenticated_request(
            "tasks/request",
            method="post",
            params={"task_type": task_type},
            data={"public_address": self.wallet.hotkey.ss58_address}
        )
        return response.json() if response and response.status_code == 200 else None

    def submit_result(self, task_type: str, batch_id: str, result: dict):
        """Submit task result back to pool"""
        endpoint = "tasks/submit_evolution" if task_type == "evolve" else "tasks/submit_evaluation"
        response = self.make_authenticated_request(
            endpoint,
            method="post",
            data={
                "batch_id": batch_id,
                **result
            }
        )
        return response and response.status_code == 200


class ChainPushDestination(PushDestination):
    def __init__(self, chain_manager):
        self.chain_manager = chain_manager

    def push(self, gene, commit_message):
        try:
            # Compute solution hash using the provided function
            solution_hash = compute_chain_hash(str(gene)+config.gene_repo)
            logging.info(f"Pushing gene {str(gene)} with hash {solution_hash}")

            # Get current block number
            current_block = self.chain_manager.subtensor.get_current_block()

            # Create solution ID
            solution_id = SolutionId(
                repo_name=config.gene_repo,
                solution_hash=solution_hash
            )

            # Store on chain
            self.chain_manager.subtensor.commit(
                self.chain_manager.wallet,
                self.chain_manager.subnet_uid,
                solution_id.to_compressed_str()
            )

            logging.info(f"Successfully pushed solution metadata to chain at block {current_block}")
            return True
        except Exception as e:
            logging.error(f"Failed to push solution metadata to chain: {str(e)}")
            return False


class HFChainPushDestination(HuggingFacePushDestination):
    def __init__(self, repo_name, chain_manager, config, **kwargs):
        super().__init__(repo_name)
        self.chain_push = ChainPushDestination(chain_manager)
        self.config = config

    def push(self, gene, commit_message, save_temp=config.Miner.save_temp_only):
        # First push to HuggingFace

        # Then push to chain
        success = self.chain_push.push(gene, commit_message)

        if success:
            logging.info("Chain push likely successful. Attempting to push to HF")
            super().push(gene, commit_message, config, save_temp)
        else:
            logging.warn("Chain push unsuccessful. Failed to push gene !")
