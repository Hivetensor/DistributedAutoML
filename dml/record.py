import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List


class GeneRecordManager:
    def __init__(self, json_file_path: str = "gene_records.json"):
        self.json_file_path = json_file_path
        self.records: Dict[str, Any] = {}
        self._load_records()

    def _load_records(self):
        if os.path.exists(self.json_file_path):
            with open(self.json_file_path, "r") as f:
                self.records = json.load(f)

    def _save_records(self):
        with open(self.json_file_path, "w") as f:
            json.dump(self.records, f, indent=2)

    def add_record(
        self, miner_hotkey: str, gene_hash: str, timestamp: float, performance: float
    ):
        self.records[miner_hotkey] = {
            "gene_hash": gene_hash,
            "timestamp": timestamp,
            "performance": performance,
        }
        self._save_records()

    def get_record(self, miner_hotkey: str) -> Dict[str, Any]:
        return self.records.get(miner_hotkey, None)

    def get_all_records(self) -> Dict[str, Dict[str, Any]]:
        return self.records

    def should_download(self, miner_hotkey: str, remote_gene_hash: str) -> bool:
        record = self.get_record(miner_hotkey)
        if record is None:
            return True
        return record["gene_hash"] != remote_gene_hash
