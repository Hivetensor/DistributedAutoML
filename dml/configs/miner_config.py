import time 

class MinerConfig:
    #TODO limit validator memory allowed to prevent DOS attacks 
    checkpoint_save_dir = "/home/mekaneeky/repos/dautoml/checkpoint/island"
    check_registration_interval = 500
    evaluation_iterations = 10
    gp_tree_height = 90
    generations = 10000
    migration_interval = 100
    migrants_per_round = 10
    miner_type = "loss"
    num_processes = 1
    pool_url = None #"http://127.0.0.1:5000"
    population_size = 200 # Per process pop = population_size // num_processes
    push_platform = "hf"
    save_temp_only = True
    seed = int(time.time())
    tournament_size = 2    
    training_iterations = 10
    
