class EvaluatorConfig:
    epochs = 1
    max_batches = 1000 #per epoch
    validate_every = 100
    
    llm_validation_steps = 100 #If not an LLM evaluates on whole val set.

    architectures = {
        "food101": [ "resnet", "convnext" ],
        "inaturalist_mini": ["resnet", "convnext"]
    }