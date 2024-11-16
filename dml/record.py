import torch
import torch.nn.functional as F
import json
import hashlib
import time
import os
from typing import Dict, Any, List
import logging
from huggingface_hub import list_repo_commits
from dml.configs.config import config
from dml.gene_io import load_individual_from_json, save_individual_to_json

def load_test_datasets():
    """Load samples from different datasets including text data for signature computation"""
    device = torch.device(config.device)
    
    # # Download required NLTK data
    # nltk.download('punkt', quiet=True)
    
    # # Simple text samples with sentiment labels (0=negative, 1=positive)
    # text_samples = [
    #     ("This movie was terrible and boring", 0),
    #     ("I loved this film, it was amazing", 1),
    #     ("The worst experience ever", 0),
    #     ("Great service and friendly staff", 1),
    # ] * 16  # Repeat to get 64 samples
    
    # # Create vocabulary from words
    # vocab = set()
    # for text, _ in text_samples:
    #     tokens = word_tokenize(text.lower())
    #     vocab.update(tokens)
    # vocab = {word: idx + 1 for idx, word in enumerate(sorted(vocab))}  # Start from 1, leave 0 for padding
    
    # Process text data
    # max_len = 20
    # text_batch = []
    # label_batch = []
    
    # for text, label in text_samples:
    #     tokens = word_tokenize(text.lower())
    #     indices = [vocab.get(token, 0) for token in tokens[:max_len]]
    #     padded = torch.tensor(indices + [0] * (max_len - len(indices)), device=device)
    #     text_batch.append(padded)
    #     label_batch.append(label)
    
    # text_batch = torch.stack(text_batch)
    # label_batch = torch.tensor(label_batch, device=device)
    
    test_inputs = {
        'basic': (
            torch.linspace(-10, 10, 100, device=device),
            torch.rand(100, device=device) * 20 - 10
        )
        # 'text': (
        #     text_batch,
        #     F.one_hot(label_batch, num_classes=2).float()
        # )
    }

    return test_inputs

class GeneRecordManager:
    def __init__(self, json_file_path: str = 'gene_records.json', expression_registry_path: str = 'expression_registry.json'):
        self.json_file_path = json_file_path
        self.expression_registry_path = expression_registry_path
        self.records: Dict[str, Any] = {}
        self.expression_registry: Dict[str, List[str]] = {}  # Maps expression hash to list of miners who used it
        self.datasets = load_test_datasets()
        self._load_records()
        self._load_expression_registry()

    def _load_records(self):
        if os.path.exists(self.json_file_path):
            with open(self.json_file_path, 'r') as f:
                self.records = json.load(f)

    def _save_records(self):
        with open(self.json_file_path, 'w') as f:
            json.dump(self.records, f, indent=2)

    def _load_expression_registry(self):
        if os.path.exists(self.expression_registry_path):
            with open(self.expression_registry_path, 'r') as f:
                self.expression_registry = json.load(f)

    def _save_expression_registry(self):
        with open(self.expression_registry_path, 'w') as f:
            try:
                json.dump(self.expression_registry, f, indent=2)
            except:
                breakpoint()

    def _compute_expression_hash(self, expr) -> str:
        """Compute a unique hash for any expression"""
        return hashlib.sha256(str(expr).encode()).hexdigest()

    def _compute_function_signature(self, func) -> str:
        """Compute a functional signature by evaluating expression on test inputs"""
        
        
        try:
            outputs = []
            for test_inputs in self.datasets.values():
                x, y = test_inputs
                
                
                # Evaluate the expression on batches of inputs
                
                #for inputs, targets in test_inputs:
                    
                try:
                    batch_result = func(x, y)
                    # Handle both scalar and tensor outputs
                    outputs.append(batch_result.detach())
                except Exception as e:
                    logging.warning(f"Error evaluating expression batch: {e}")
                    return None

            # Concatenate all outputs
            if len(outputs) > 1:
                outputs = torch.cat(outputs)
            elif len(outputs)==1:
                outputs = outputs[0]

            # Create a hash from the rounded outputs
            # Round to 6 decimal places to handle floating point differences
            rounded_outputs = torch.round(outputs * 1e6) / 1e6
            output_bytes = rounded_outputs.cpu().numpy().tobytes()
                
            return hashlib.sha256(output_bytes).hexdigest()
                    
        except Exception as e:
            logging.error(f"Failed to compute function signature: {e}")
            return None

    def add_record(self, 
                  miner_hotkey: str, 
                  gene_hash: str, 
                  timestamp: float, 
                  performance: float, 
                  expr=None, 
                  repo_name: str = None, 
                  func = None):
        """Add a new record for a miner's gene submission"""

        try:
            performance = performance.tolist()
        except:
            pass
        
        self.records[miner_hotkey] = {
            'gene_hash': gene_hash,
            'timestamp': timestamp,
            'performance': performance,
            'expr': str(expr) if expr else None
        }
        
        
        if expr is not None:
            func_signature = self._compute_function_signature(func)
            if func_signature:
                if func_signature not in self.expression_registry:

                    self.expression_registry[func_signature] = {
                        "earliest_timestamp": timestamp,
                        "earliest_hotkey": miner_hotkey,
                        "score": performance
                    }
                    self._save_expression_registry()
            #if func_signature is None?
        
        
        self._save_records()

    def is_expression_duplicate(self, expr) -> bool:
        """Check if an expression has been used before by any miner"""
        expr_fingerprint = self._compute_function_signature(expr)
        return expr_fingerprint in self.expression_registry


    def get_record(self, miner_hotkey: str) -> Dict[str, Any]:
        return self.records.get(miner_hotkey, None)

    def get_all_records(self) -> Dict[str, Dict[str, Any]]:
        return self.records

    def should_download(self, miner_hotkey: str, remote_gene_hash: str) -> bool:
        record = self.get_record(miner_hotkey)
        if record is None:
            return True
        return record['gene_hash'] != remote_gene_hash