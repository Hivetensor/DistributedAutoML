import time 

class MinerConfig:
    #TODO limit validator memory allowed to prevent DOS attacks 
    device = "cuda"
    batch_size = 8
    checkpoint_save_dir = "checkpoints"
    check_registration_interval = 500
    evaluation_iterations = 50
    gp_tree_height = 90
    generations = 10000000000 
    miner_type = "loss"
    miner_operation="evolve" #evolve/evaluate
    num_processes = 4
    num_processes = 1 #TODO add multi-arch multi-core mining support
    pool_url = None #"http://127.0.0.1:5000"
    population_size = 50 # Per process pop = population_size // num_processes
    push_platform = "hf"
    save_temp_only = True
    seed = int(time.time())
    tournament_size = 2    
    training_iterations = 3
    dataset_names = ["mnist"]
    training_iterations = 100
    architectures = {
        "cifar10": [ "mlp" ],
        "imagenette": [ "resnet", "mobilenet_v3", "efficientnet_v2" ],
        "flowers102": ["resnet", "mobilenet_v3", "efficientnet_v2"]
    }
    architectures_weights = {
        "mlp":0.25,
        "resnet":0.25,
        "efficientnet_v2":0.25,
        "mobilenet_v3":0.25,
    }


