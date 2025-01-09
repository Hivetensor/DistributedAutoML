import bittensor as bt

class BittensorConfig:
    netuid = 49
    wallet_name = "your_wallet"
    wallet_hotkey = "your_hotkey"
    path = "~/.bittensor/wallets/"
    network = "finney"  # or "finney" for mainnet
    epoch_length = 100
    #subtensor_chain_endpoint = bt.__finney_entrypoint__ #"ws://127.0.0.1:9944"  # local subtensor

    @classmethod
    def get_bittensor_config(cls):
        bt_config = bt.config()
        bt_config.wallet = bt.config()
        bt_config.subtensor = bt.config()
        bt_config.netuid = cls.netuid
        bt_config.wallet.name = cls.wallet_name
        bt_config.wallet.hotkey = cls.wallet_hotkey
        bt_config.subtensor.network = cls.network
        bt_config.epoch_length = cls.epoch_length
        #bt_config.subtensor.chain_endpoint = cls.subtensor_chain_endpoint
        return bt_config
