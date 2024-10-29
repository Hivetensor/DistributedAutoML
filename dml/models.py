import torch.nn as nn
import torch 
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class MNISTModel(nn.Module):
    def __init__(self, activation_fn):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.activation_fn = activation_fn
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_fn(self.fc2(x))
        return self.fc3(x)

class CIFARModel(nn.Module):
    def __init__(self, activation_fn):
        super().__init__()
        self.conv1 = ConvBlock(3, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)
        self.activation_fn = activation_fn
        
    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.flatten(x)
        x = self.dropout(self.activation_fn(self.fc1(x)))
        return self.fc2(x)

class ShakespeareModel(nn.Module):
    def __init__(self, activation_fn, vocab_size=50257, embed_size=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, 512, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, vocab_size)
        self.dropout = nn.Dropout(0.2)
        self.activation_fn = activation_fn
        
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.activation_fn(self.fc1(x))
        return self.fc2(x)

class ImageNetModel(nn.Module):
    def __init__(self, activation_fn):
        super().__init__()
        self.conv1 = ConvBlock(3, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 = ConvBlock(256, 512)
        self.conv5 = ConvBlock(512, 512)
        self.pool = nn.MaxPool2d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)
        self.dropout = nn.Dropout(0.5)
        self.activation_fn = activation_fn
        
    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(self.conv4(x))
        x = self.pool(self.conv5(x))
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.dropout(self.activation_fn(self.fc1(x)))
        x = self.dropout(self.activation_fn(self.fc2(x)))
        return self.fc3(x)

class ModelFactory:
    @staticmethod
    def create_model(model_type: str, activation_fn, **kwargs):
        """
        Factory method to create models based on type
        
        Args:
            model_type: String identifier for the model type
            activation_fn: The evolved activation function to use
            **kwargs: Additional arguments specific to each model type
        """
        models = {
            'mnist': lambda: MNISTModel(activation_fn),
            'cifar10': lambda: CIFARModel(activation_fn),
            'shakespeare': lambda: ShakespeareModel(activation_fn, **kwargs),
            'imagenet': lambda: ImageNetModel(activation_fn)
        }
        
        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")
            
        return models[model_type]()