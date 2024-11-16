from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import logging
from typing import Dict, Any
import time
import torch
from dml.record import GeneRecordManager
from dml.gene_io import load_individual_from_json
from dml.ops import create_pset_validator
from dml.chain.btt_connector import BittensorNetwork
from dml.configs.config import config

from deap import base, creator, gp, tools
from substrateinterface import Keypair, KeypairType

from fastapi import FastAPI, HTTPException
import logging
from typing import Dict, Any
import time
import torch
from dml.record import GeneRecordManager
from dml.gene_io import load_individual_from_json
from dml.ops import create_pset_validator
from deap import base, creator, gp, tools
from substrateinterface import Keypair, KeypairType
from fastapi import Request

import json

app = FastAPI()
logging.basicConfig(level=logging.INFO)

class ValidatorServer:
    def __init__(self, metagraph):
        self.gene_record_manager = GeneRecordManager()
        self.metagraph = metagraph
        self.setup_toolbox()
        
    def setup_toolbox(self):
        self.toolbox = base.Toolbox()
        self.pset = create_pset_validator()
        
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
            
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=3)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("compile", gp.compile, pset=self.pset)

    def verify_miner(self, public_address: str, message: str, signature: str) -> bool:
        # Check if miner exists in metagraph
        try:
            uid = self.metagraph.hotkeys.index(public_address)
        except ValueError:
            return False
            
        # Verify signature
        try:
            signature_bytes = bytes.fromhex(signature) if isinstance(signature, str) else signature
            keypair = Keypair(ss58_address=public_address, crypto_type=KeypairType.SR25519)
            return keypair.verify(message.encode('utf-8'), signature_bytes)
        except Exception as e:
            logging.error(f"Signature verification failed: {e}")
            return False

server = None

@app.post("/push_gene")
async def submit_gene(request: Request):
    try:
        data = await request.json()
        
        # Extract authentication fields from whatever format they come in
        public_address = data.get('public_address') or data.get('hotkey')
        message = data.get('message', '')
        signature = data.get('signature', '')
        gene_data = json.loads(data.get('gene'))
        
        
        if not all([public_address, message, signature, gene_data]):
            raise HTTPException(status_code=400, detail="Missing required fields")
            
        if not server.verify_miner(public_address, message, signature):
            raise HTTPException(status_code=403, detail="Invalid authentication")

        gene, compiled_function = load_individual_from_json(data=gene_data,toolbox=server.toolbox, pset=server.pset)
        #compiled_gene = server.toolbox.compile(expr=gene)
        
        if server.gene_record_manager.is_expression_duplicate(compiled_function):
            raise HTTPException(status_code=400, detail="Duplicate gene detected. Was previously submitted by another miner.")
            
        server.gene_record_manager.add_record(
            miner_hotkey=public_address,
            gene_hash=None,
            timestamp=time.time(),
            performance=None,
            expr=gene,
            func=compiled_function
        )
        
        return {"status": "success", "message": "Gene submitted for evaluation"}
        
    except Exception as e:
        logging.error(f"Error processing gene submission: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

def run_server(metagraph, host="0.0.0.0", port=8000):
    global server
    server = ValidatorServer(metagraph)
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    bt_config = config.get_bittensor_config()
    BittensorNetwork.initialize(bt_config)
    config.bittensor_network = BittensorNetwork

    run_server(config.bittensor_network.metagraph)


