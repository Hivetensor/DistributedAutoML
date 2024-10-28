from dml.configs.bittensor_config import BittensorConfig
from dml.configs.general_config import GeneralConfig
from dml.configs.miner_config import MinerConfig
from dml.configs.validator_config import ValidatorConfig


class Config:
    def __init__(self):
        self.general = GeneralConfig()
        self.Bittensor = BittensorConfig()
        self.Miner = MinerConfig()
        self.Validator = ValidatorConfig()

    @property
    def device(self):
        return self.general.device

    @property
    def hf_token(self):
        return self.general.hf_token

    @property
    def gene_repo(self):
        return self.general.gene_repo

    @property
    def metrics_file(self):
        return self.general.metrics_file

    def get_bittensor_config(self):
        return self.Bittensor.get_bittensor_config()


config = Config()
