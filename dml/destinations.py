import base64
import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# import arweave
import requests
from huggingface_hub import HfApi

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class PushDestination(ABC):
    @abstractmethod
    def push(self, gene: Dict[str, Any], commit_message: str) -> bool:
        pass

    @abstractmethod
    def verify(self, gene_data: Dict[str, Any]) -> bool:
        pass


class PushMixin:
    def __init__(self):
        self.push_destinations = []

    def push_to_remote(self, gene: Dict[str, Any], commit_message: str) -> None:
        if not hasattr(self, "push_destinations"):
            logging.warning("No push destinations defined")
            return

        for dest in self.push_destinations:
            try:
                success = dest.push(gene, commit_message)
                if not success:
                    logging.error(f"Push failed for {dest.__class__.__name__}")
            except Exception as e:
                logging.error(f"Error in {dest.__class__.__name__}: {e}")


class StorageBase(PushDestination):
    def _prepare_metadata(self, commit_message: str) -> Dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "commit_message": commit_message,
            "version": "1.0",
            "storage_type": self.__class__.__name__,
        }

    def _extract_fitness(self, commit_message: str) -> Optional[float]:
        try:
            if "fitness=" in commit_message:
                return float(commit_message.split("fitness=")[-1].split()[0])
            return None
        except (ValueError, IndexError):
            return None

    def _prepare_content(
        self, gene: Dict[str, Any], commit_message: str
    ) -> Dict[str, Any]:
        return {
            "gene": gene,
            "metadata": self._prepare_metadata(commit_message),
            "fitness": self._extract_fitness(commit_message),
        }


class HuggingFacePushDestination(StorageBase):
    def __init__(self, repo_name: str, token: str):
        self.repo_name = repo_name
        self.api = HfApi(token=token)

    def push(self, gene: Dict[str, Any], commit_message: str) -> bool:
        if not self.repo_name:
            logging.info("No HuggingFace repository provided")
            return False

        content = self._prepare_content(gene, commit_message)

        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".json"
        ) as temp_file:
            json.dump(content, temp_file)
            temp_path = temp_file.name

        try:
            self.api.upload_file(
                path_or_fileobj=temp_path,
                path_in_repo="best_gene.json",
                repo_id=self.repo_name,
                commit_message=commit_message,
            )
            return self.verify(content)
        finally:
            os.unlink(temp_path)

    def verify(self, gene_data: Dict[str, Any]) -> bool:
        try:
            url = f"https://huggingface.co/{self.repo_name}/raw/main/best_gene.json"
            response = requests.get(url)
            if response.status_code != 200:
                return False
            stored_data = response.json()
            return stored_data == gene_data
        except Exception as e:
            logging.error(f"Verification failed: {e}")
            return False


class GitHubPushDestination(StorageBase):
    def __init__(self, repo_owner: str, repo_name: str, token: str):
        self.api_base = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }

    def push(self, gene: Dict[str, Any], commit_message: str) -> bool:
        content = self._prepare_content(gene, commit_message)

        current = self._get_current_file()
        sha = current["sha"] if current else None

        try:
            self._push_to_github(content, sha)
            return self.verify(content)
        except Exception as e:
            logging.error(f"GitHub push failed: {e}")
            return False

    def verify(self, gene_data: Dict[str, Any]) -> bool:
        current = self._get_current_file()
        if not current:
            return False
        stored_content = json.loads(base64.b64decode(current["content"]).decode())
        return stored_content == gene_data

    def _get_current_file(self) -> Optional[Dict[str, Any]]:
        response = requests.get(
            f"{self.api_base}/contents/best_gene.json", headers=self.headers
        )
        return response.json() if response.status_code == 200 else None

    def _push_to_github(self, content: Dict[str, Any], sha: Optional[str]) -> None:
        data = {
            "message": content["metadata"]["commit_message"],
            "content": base64.b64encode(json.dumps(content).encode()).decode(),
        }
        if sha:
            data["sha"] = sha

        response = requests.put(
            f"{self.api_base}/contents/best_gene.json", headers=self.headers, json=data
        )

        if response.status_code not in (200, 201):
            raise Exception(f"GitHub API error: {response.text}")


# TODO: Explore arweave as potential option
# class ArweavePushDestination(StorageBase):
#     def __init__(self, wallet_file: str):
#         self.wallet = arweave.Wallet(wallet_file)
#
#     def push(self, gene: Dict[str, Any], commit_message: str) -> bool:
#         content = self._prepare_content(gene, commit_message)
#
#         try:
#             transaction = arweave.Transaction.create(
#                 self.wallet,
#                 data=json.dumps(content)
#             )
#             transaction.add_tag('Content-Type', 'application/json')
#             transaction.add_tag('Type', 'automl-gene')
#             transaction.sign(self.wallet)
#             transaction.post()
#
#             return self.verify(content)
#         except Exception as e:
#             logging.error(f"Arweave push failed: {e}")
#             return False
#
#     def verify(self, gene_data: Dict[str, Any]) -> bool:
#         # Basic implementation - would need proper Arweave verification
#         return True


class MultiDestinationPush(PushMixin):
    def __init__(self, destinations: List[PushDestination]):
        super().__init__()
        self.push_destinations = destinations
