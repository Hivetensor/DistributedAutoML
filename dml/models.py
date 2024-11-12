import torch.nn as nn
import torch 

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

def get_mnist_model(
    hidden_size: int = 128,
    dropout: float = 0.2
) -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, 10)
    )

def get_cifar_model(
    hidden_size: int = 256,
    dropout: float = 0.2
) -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3072, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, 10)
    )

def get_cifar100_model(
    hidden_size: int = 512,
    dropout: float = 0.3
) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(128, 256, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(256 * 4 * 4, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, 100)
    )

# Dictionary mapping dataset names to their model creators
MODEL_CREATORS = {
    'mnist': get_mnist_model,
    'cifar10': get_cifar_model,
    'cifar100': get_cifar100_model,
    'imagenet': get_imagenet_model,
    'shakespeare': get_shakespeare_model

}

def get_model_for_dataset(dataset_name: str, **kwargs) -> nn.Module:
    """Get the appropriate model for a given dataset"""
    if dataset_name not in MODEL_CREATORS:
        raise ValueError(f"No model creator found for dataset: {dataset_name}")
    
    return MODEL_CREATORS[dataset_name](**kwargs)

