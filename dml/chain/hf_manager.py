import hashlib
import os
import subprocess

import torch
from bittensor import logging
from dotenv import load_dotenv
from huggingface_hub import HfApi, HfFolder, Repository, hf_hub_download, scan_cache_dir

load_dotenv()


class HFManager:
    """
    Manages interactions with the Hugging Face Hub for operations such as cloning, pushing and pulling models or weights/gradients.
    """

    def __init__(
        self,
        local_dir=".",  # gradients local
        hf_token=os.getenv("HF_TOKEN"),
        gene_repo_id=None,  # gradients HF
        model_dir=None,  # averaged local
        device="cuda",
    ):
        # Initializes the HFManager with the necessary repository and authentication details.
        self.gene_repo_id = gene_repo_id

        self.hf_token = hf_token
        self.device = device
        # self.local_dir = local_dir

        # Define the local directory structure based on repository IDs but only do clone personal repo if miner
        if self.gene_repo_id != None:
            self.gene_repo = Repository(
                local_dir=os.path.join(local_dir, gene_repo_id.split("/")[-1]),
                clone_from=gene_repo_id,
                use_auth_token=hf_token,
            )
            self.local_gene_dir = os.path.join(local_dir, gene_repo_id.split("/")[-1])

        self.api = HfApi()
        # Get the latest commit SHA for synchronization checks
        self.latest_gene_commit_sha = self.get_latest_commit_sha(self.gene_repo_id)

    @staticmethod
    def clear_hf_cache():
        # Get the cache directory
        hf_cache_info = scan_cache_dir()
        commit_hashes = [
            revision.commit_hash
            for repo in hf_cache_info.repos
            for revision in repo.revisions
        ]

        # Check if the cache directory exists
        delete_strategy = scan_cache_dir().delete_revisions(*commit_hashes)

        logging.info("Will free " + delete_strategy.expected_freed_size_str)
        delete_strategy.execute()

    @staticmethod
    def git_prune_and_refresh(repo_path):
        """
        Change to the specified repository directory, execute 'git lfs prune', and revert to the original directory.
        """
        original_dir = os.getcwd()
        try:
            os.chdir(repo_path)
            subprocess.run(["git", "config", "pull.rebase", "true"], check=True)
            subprocess.run(["git", "pull", "--force"], check=True)
            subprocess.run(["git", "lfs", "prune"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to prune Git LFS objects: {e}")
        finally:
            os.chdir(original_dir)

    def push_changes(self, file_to_send):
        """
        Stages, commits, squashes, and pushes changes to the configured repository.
        Also prunes unnecessary Git LFS objects to free up storage.
        """
        try:
            # Stage the changes
            self.gradient_repo.git_add(file_to_send)

            # Commit with a unified message
            self.gradient_repo.git_commit("Squashed commits - update model gradients")

            # Push the changes to the repository
            self.gradient_repo.git_push()

            self.api.super_squash_history(repo_id=self.my_repo_id)

            # Prune unneeded Git LFS objects and pull the squashed version locally
            self.git_prune_and_refresh(
                self.local_gradient_dir
            )  # Clean up unused LFS objects

        except Exception as e:
            print(f"Failed to push changes: {e}")

    def push_to_hf_hub(self, path_to_model, commit_message="Pushing model to Hub"):
        try:
            # Stage the changes
            self.model_repo.git_add(path_to_model)

            # Squash commits into a single one before pushing

            # Commit with a unified message
            self.model_repo.git_commit("Squashed commits - update model gradients")

            self.model_repo.git_push()

            self.api.super_squash_history(repo_id=self.model_repo_id)

            # Prune unneeded Git LFS objects and pull the squashed version locally
            self.git_prune_and_refresh(self.model_dir)

            # Push the changes to the repository

        except Exception as e:
            print(f"Failed to push changes: {e}")

    def get_latest_commit_sha(self, repo):
        """
        Fetches the latest commit SHA of the specified repository from the Hugging Face Hub.
        """
        try:
            repo_info = self.api.repo_info(repo)
            latest_commit_sha = repo_info.sha
            # print(latest_commit_sha)
            return latest_commit_sha
        except Exception as e:
            logging.info(f"Failed to fetch latest commit SHA: {e}")
            return None

    def check_for_new_submissions(self, repo):
        """
        Compares the current commit SHA with the latest to determine if there are new submissions.
        """
        current_commit_sha = self.get_latest_commit_sha(repo)
        if current_commit_sha != self.latest_model_commit_sha:
            self.latest_model_commit_sha = current_commit_sha
            return True
        return False

    def update_model(self, model, model_file_name="averaged_model.pt"):
        """
        Loads an updated model from a .pt file and updates the in-memory model's parameters.
        """
        model_path = os.path.join(self.model_dir, model_file_name)
        if os.path.exists(model_path):
            model_state_dict = torch.load(model_path)
            model.load_state_dict(model_state_dict)
            model.train()
            logging.info(f"Model updated from local path: {model_path}")
            return model
        else:
            raise FileNotFoundError(f"{model_file_name} not found in the repository.")

    def get_local_gradient_directory(self):
        """Return the local directory of the repository."""
        return self.local_gradient_dir

    def get_local_model_directory(self):
        """Return the local directory of the repository."""
        return self.model_dir

    def pull_latest_model(self):
        self.model_repo.git_pull()

    def receive_gradients(self, miner_repo_id, weights_file_name="weight_diff.pt"):
        try:  # TODO Add some garbage collection.
            # Download the gradients file from Hugging Face Hub
            weights_file_path = hf_hub_download(
                repo_id=miner_repo_id, filename=weights_file_name, use_auth_token=True
            )
            # Load the gradients directly using torch.load
            miner_weights = torch.load(weights_file_path, map_location=self.device)
            os.remove(weights_file_path)
            return miner_weights
        except Exception as e:
            logging.debug(f"Error receiving gradients from Hugging Face: {e}")
