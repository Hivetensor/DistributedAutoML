class EvaluatorConfig:
    epochs = 1
    max_batches = 1000 #per epoch
    validate_every = 100
    
    llm_validation_steps = 100 #If not an LLM evaluates on whole val set.

    architectures = {
        "cifar10": [ "mlp" ],
        "imagenette": [ "resnet", "mobilenet_v3", "efficientnet_v2" ],
        "fgvc_aircraft": ["resnet", "mobilenet_v3", "efficientnet_v2"]
    }