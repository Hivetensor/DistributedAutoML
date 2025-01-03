import torch.nn as nn
import torch 
from torch.optim import Optimizer
import torchvision.models as models
from dml.data import load_datasets, dataset_configs



class BaselineNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class EvolvableNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, evolved_activation):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.evolved_activation = evolved_activation

    def forward(self, x):
        x = x.view(-1,28*28)
        x = self.fc1(x)
        x = self.evolved_activation(x)
        x = self.fc2(x)
        return x
    
class EvolvedLoss(torch.nn.Module):
    def __init__(self, genome, device = "cpu"):
        super().__init__()
        self.genome = genome
        self.device = device

    def forward(self, outputs, targets):
        #outputs = outputs.detach().float().requires_grad_()#.to(self.device)
        #targets = targets.detach().float().requires_grad_()#.to(self.device)
        
        memory = self.genome.memory
        memory.reset()
        
        memory[self.genome.input_addresses[0]] = outputs
        memory[self.genome.input_addresses[1]] = targets
        for i, op in enumerate(self.genome.gene):
            func = self.genome.function_decoder.decoding_map[op][0]
            input1 = memory[self.genome.input_gene[i]]#.to(self.device)
            input2 = memory[self.genome.input_gene_2[i]]#.to(self.device)
            constant = torch.tensor(self.genome.constants_gene[i], requires_grad=True)#.to(self.device)
            constant_2 = torch.tensor(self.genome.constants_gene_2[i], requires_grad=True)#.to(self.device)
            
            output = func(input1, input2, constant, constant_2, self.genome.row_fixed, self.genome.column_fixed)
            if output is not None:
                memory[self.genome.output_gene[i]] = output

        loss = memory[self.genome.output_addresses[0]]
        return loss
    
class BabyGPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, num_layers=2, num_heads=2, sequence_length=512, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(sequence_length, embedding_dim)
        
        transformer_blocks = []
        for _ in range(num_layers):
            transformer_blocks.append(nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout
            ))
        self.transformer = nn.TransformerEncoder(
            encoder_layer=transformer_blocks[0],
            num_layers=len(transformer_blocks)
        )
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, x):
        B, T = x.shape
        
        mask = self.generate_square_subsequent_mask(T).to(x.device)
        
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        x = tok_emb + pos_emb
        
        x = x.transpose(0, 1)  # TransformerEncoder expects seq_len first
        x = self.transformer(x, mask=mask)  # 
        x = x.transpose(0, 1)  # Back to batch first
        
        logits = self.fc_out(x)
        return logits


def get_imagenet_model(
    num_classes: int = 1000,
    pretrained: bool = False
) -> nn.Module:
    """
    Returns a ResNet50 model suitable for ImageNet
    """
    if pretrained:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

def get_shakespeare_model(
    embed_size: int = 384,
    num_heads: int = 6,
    num_layers: int = 6,
    **kwargs
) -> nn.Module:
    """
    Returns a small GPT model suitable for Shakespeare text
    """
    return BabyGPT(
        vocab_size=85,
        embedding_dim=embed_size,
        num_heads=num_heads,
        num_layers=num_layers
    )

def get_mlp(input_size: int, output_size: int, hidden_size: int = 128, dropout: float = 0.2) -> nn.Module:
    """Generic MLP that works with any dataset dimensions."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, output_size)
    )

def get_cnn(input_channels: int, output_size: int, base_channels: int = 32) -> nn.Module:
    """Generic CNN that works with any image dataset."""
    return nn.Sequential(
        nn.Conv2d(input_channels, base_channels, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(base_channels * 2 * 7 * 7, 128),  # This assumes 28x28 input, would need adjustment
        nn.ReLU(),
        nn.Linear(128, output_size)
    )

def get_baby_gpt(vocab_size: int, embed_size: int = 384, num_heads: int = 6, num_layers: int = 6) -> nn.Module:
    """Generic GPT model that works with any text dataset."""
    return BabyGPT(
        vocab_size=vocab_size,
        embedding_dim=embed_size,
        num_heads=num_heads,
        num_layers=num_layers
    )

def get_mobilenet_v3_large(
    num_classes: int = 1000,
    pretrained: bool = True
) -> nn.Module:
    """
    Returns a MobileNetV3-Large model with optional pretrained weights
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights from ImageNet
    
    Returns:
        nn.Module: MobileNetV3-Large model
    """
    if pretrained:
        model = models.mobilenet_v3_large(weights='IMAGENET1K_V2')
        if num_classes != 1000:
            # Replace the classifier for different number of classes
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    
    return model

def get_efficientnet_v2_m(
    num_classes: int = 1000,
    pretrained: bool = False
) -> nn.Module:
    """
    Returns an EfficientNetV2-M model
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        nn.Module: EfficientNetV2-M model
    """
    if pretrained:
        model = models.efficientnet_v2_m(weights='IMAGENET1K_V2')
        if num_classes != 1000:
            # Replace the classifier for different number of classes
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        model = models.efficientnet_v2_m(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    
    return model

def get_convnext(num_classes: int = 1000, pretrained: bool = True) -> nn.Module:
   """Returns ConvNeXt model"""
   if pretrained:
       model = models.convnext_base(weights='IMAGENET1K_V1')
       if num_classes != 1000:
           model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
   else:
       model = models.convnext_base(weights=None)
       model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
   return model

def get_alexnet(num_classes: int = 1000, pretrained: bool = True) -> nn.Module:
   """Returns AlexNet model"""
   if pretrained:
       model = models.alexnet(weights='IMAGENET1K_V1')
       if num_classes != 1000:
           model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
   else:
       model = models.alexnet(weights=None)
       model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
   return model


# Dictionary mapping dataset names to their model creators
ARCHITECTURE_MAP = {
    'mlp': get_mlp,
    'cnn': get_cnn,
    'gpt': get_baby_gpt,
    'resnet': get_imagenet_model,  # Already generalized in original code
    'mobilenet_v3': get_mobilenet_v3_large,
    'efficientnet_v2': get_efficientnet_v2_m,
    'convnext': get_convnext,
   'alexnet': get_alexnet
}




def get_model_for_dataset(dataset_name: str, architecture: str = 'mlp', dataset_spec = None, **kwargs) -> nn.Module:
    """Get the appropriate model for a given dataset and architecture.
    
    Args:
        dataset_name: Name of the dataset
        architecture: Name of the architecture ('mlp', 'cnn', 'gpt', 'resnet')
        dataset_spec: DatasetSpec object containing dataset parameters
        **kwargs: Additional arguments to pass to the model creator
        
    Returns:
        nn.Module: The initialized model
    """
    if architecture not in ARCHITECTURE_MAP:
        raise ValueError(f"Architecture {architecture} not recognized")
    
    if dataset_spec is None:
        # Get dataset spec if not provided
        dataset_spec = dataset_configs[dataset_name]
    
    # Get the appropriate model creator
    model_creator = ARCHITECTURE_MAP[architecture]
    
    # Configure architecture-specific parameters
    if architecture == 'mlp':
        return model_creator(
            input_size=dataset_spec["input_size"],
            output_size=dataset_spec["output_size"],
            **kwargs
        )
    elif architecture == 'cnn':
        # Assume image data with channels
        if isinstance(dataset_spec["input_size"], tuple):
            input_channels = dataset_spec["input_size"][0]
        else:
            input_channels = 1  # Default to single channel
        return model_creator(
            input_channels=input_channels,
            output_size=dataset_spec["output_size"],
            **kwargs
        )
    elif architecture == 'gpt':
        return model_creator(
            vocab_size=dataset_spec["output_size"],
            embed_size=dataset_spec["hidden_size"],
            **kwargs
        )
    else:  # resnet or other architectures
        return model_creator(
            num_classes=dataset_spec["output_size"],
            **kwargs
        )


class TorchEvolvedOptimizer(Optimizer):
   def __init__(self, params, evolved_func, lr=1e-3, weight_decay=0):
       defaults = dict(lr=lr, weight_decay=weight_decay)
       super().__init__(params, defaults)
       self.evolved_func = evolved_func
       self.state_size = 32 # Size of state vector
       # Initialize state vector for each param group
       self.optimizer_states = {i: torch.zeros(self.state_size) 
                              for i in range(len(self.param_groups))}

   def step(self, closure=None):
       loss = None
       if closure is not None:
           loss = closure()

       for group_idx, group in enumerate(self.param_groups):
           for p in group['params']:
               if p.grad is None:
                   continue
                   
               # Get state for this param group
               state = self.optimizer_states[group_idx]
               
               # Run evolved function
               param_update, new_state = self.evolved_func(
                   p.data,
                   p.grad.data,
                   state,
                   group['lr'],
                   group['weight_decay']
               )
               
               # Store new state
               self.optimizer_states[group_idx] = new_state
               
               # Apply updates
               p.data.add_(param_update)
               
       return loss

class ModelArchitectureSpec:
    def __init__(self, name: str, model_fn, input_size: int, output_size: int):
        self.name = name
        self.model_fn = model_fn
        self.input_size = input_size
        self.output_size = output_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def create_model(self):
        return self.model_fn().to(self.device)