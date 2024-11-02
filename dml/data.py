from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import Any, List, Tuple, Optional

@dataclass
class DatasetSpec:
    name: str
    input_size: int
    output_size: int
    hidden_size: int = 128
    train_loader: Optional[DataLoader] = None
    val_loader: Optional[DataLoader] = None
    learning_rate: float = 0.001
    batch_size: int = 32
    training_iterations: int = 1
    weight: float = 1.0  

def get_mnist_loaders(
    batch_size: int = 32,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        './data', 
        train=True, 
        download=True,
        transform=transform
    )
    
    val_dataset = datasets.MNIST(
        './data', 
        train=False,
        transform=transform
    )
    
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )

def get_cifar10_loaders(
    batch_size: int = 32,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(
        './data', 
        train=True,
        download=True, 
        transform=transform
    )
    
    val_dataset = datasets.CIFAR10(
        './data', 
        train=False,
        transform=transform
    )
    
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )

def get_cifar100_loaders(
    batch_size: int = 32,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408), 
            (0.2675, 0.2565, 0.2761)
        )
    ])
    
    train_dataset = datasets.CIFAR100(
        './data',
        train=True,
        download=True,
        transform=transform
    )
    
    val_dataset = datasets.CIFAR100(
        './data',
        train=False,
        transform=transform
    )
    
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )

def get_imagenet_1k_loaders(
    data_dir: str = './data/imagenet',
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224
) -> Tuple[DataLoader, DataLoader]:
    """
    ImageNet-1K data loaders with standard augmentation
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    train_dataset = datasets.ImageNet(
        data_dir,
        split='train',
        transform=train_transform
    )
    
    val_dataset = datasets.ImageNet(
        data_dir,
        split='val',
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

class ShakespeareDataset(Dataset):
    def __init__(
        self,
        text_path: str,
        seq_length: int = 256,
        train: bool = True
    ):
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Create character level dictionary
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        data = torch.tensor([self.char_to_idx[c] for c in text], dtype=torch.long)
        
        # Split into train/val (90/10)
        n = int(0.9 * len(data))
        self.data = data[:n] if train else data[n:]
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data) - self.seq_length - 1
        
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_length + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
    



def get_shakespeare_loaders(
    text_path: str = './data/shakespeare.txt',
    batch_size: int = 64,
    seq_length: int = 256,
    num_workers: int = 1
) -> Tuple[DataLoader, DataLoader]:
    """
    Shakespeare dataset loaders for character-level language modeling
    """
    train_dataset = ShakespeareDataset(
        text_path=text_path,
        seq_length=seq_length,
        train=True
    )
    
    val_dataset = ShakespeareDataset(
        text_path=text_path,
        seq_length=seq_length,
        train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, train_dataset.vocab_size