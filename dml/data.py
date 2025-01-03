from dataclasses import dataclass
from dml.utils import set_seed
import requests
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import datasets, transforms
from typing import Any, List, Tuple, Optional, Union
import os 
import requests
import tarfile
import numpy as np
import random 

def download_imagenette():
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
    if not os.path.exists('./data/imagenette2'):
        response = requests.get(url, stream=True)
        with open('./data/imagenette2.tgz', 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        with tarfile.open('./data/imagenette2.tgz', 'r:gz') as tar:
            tar.extractall(path='./data')
        
        os.remove('./data/imagenette2.tgz')

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

class DeterministicSampler(Sampler):
    """
    Sampler for a single shuffle at initialization to ensure consistency across validators
    """
    def __init__(self, n):
        self.idx = torch.randperm(n).tolist()

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_mnist_loaders(batch_size: int = 32, num_workers: int = 2, seed: int = 42) -> Tuple[DataLoader, DataLoader]:
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
    set_seed(seed)
    train_sampler = DeterministicSampler(len(train_dataset))
    val_sampler = DeterministicSampler(len(val_dataset))
    
    return (
        DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            worker_init_fn=seed_worker,
            sampler=train_sampler
        ),
        DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            worker_init_fn=seed_worker,
            sampler=val_sampler
        )
    )

def get_cifar10_loaders(batch_size: int = 32, num_workers: int = 2, seed: int = 42) -> Tuple[DataLoader, DataLoader]:
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
    set_seed(seed)
    train_sampler = DeterministicSampler(len(train_dataset))
    val_sampler = DeterministicSampler(len(val_dataset))
    
    return (
        DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            worker_init_fn=seed_worker,
            sampler=train_sampler
        ),
        DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            worker_init_fn=seed_worker,
            sampler=val_sampler
        )
    )

def get_cifar100_loaders(batch_size: int = 32, num_workers: int = 2, seed: int = 42) -> Tuple[DataLoader, DataLoader]:
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
    set_seed(seed)
    train_sampler = DeterministicSampler(len(train_dataset))
    val_sampler = DeterministicSampler(len(val_dataset))
    
    return (
        DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            worker_init_fn=seed_worker,
            sampler=train_sampler
        ),
        DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            worker_init_fn=seed_worker,
            sampler=val_sampler
        )
    )

def get_imagenet_1k_loaders(
    data_dir: str = './data/imagenet',
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    seed: int = 42,
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
    set_seed(seed)
    train_sampler = DeterministicSampler(len(train_dataset))
    val_sampler = DeterministicSampler(len(val_dataset))
    
    return (
        DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            worker_init_fn=seed_worker,
            sampler=train_sampler
        ),
        DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            worker_init_fn=seed_worker,
            sampler=val_sampler
        )
    )

class ShakespeareDataset(Dataset):
    def __init__(
        self,
        text_path: str,
        seq_length: int = 256,
        train: bool = True
    ):
        text_url = "https://gist.githubusercontent.com/blakesanie/dde3a2b7e698f52f389532b4b52bc254/raw/76fe1b5e9efcf0d2afdfd78b0bfaa737ad0a67d3/shakespeare.txt"
        if not os.path.exists(text_path):
            response = requests.get(text_url)
            with open(text_path, "wb") as file:
                file.write(response.content)

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
    num_workers: int = 1,
    seed: int = 42,
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
    set_seed(seed)
    train_sampler = DeterministicSampler(len(train_dataset))
    val_sampler = DeterministicSampler(len(val_dataset))
    
    train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            worker_init_fn=seed_worker,
            sampler=train_sampler
    )
    
    val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            worker_init_fn=seed_worker,
            sampler=val_sampler
    )
    
    return train_loader, val_loader, train_dataset.vocab_size

def get_imagenette_loaders(
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Imagenette dataset loaders with standard augmentation.
    Imagenette is a subset of 10 easily classified classes from ImageNet.
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
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
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    download_imagenette()

    # Imagenette uses ImageFolder since it's structured like ImageNet
    train_dataset = datasets.ImageFolder(
        './data/imagenette2/train',
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        './data/imagenette2/val',
        transform=val_transform
    )
    set_seed(seed)
    train_sampler = DeterministicSampler(len(train_dataset))
    val_sampler = DeterministicSampler(len(val_dataset))
    
    return (
        DataLoader(train_dataset, batch_size=batch_size,  num_workers=num_workers, worker_init_fn=seed_worker,
            sampler=train_sampler),
        DataLoader(val_dataset, batch_size=batch_size,  num_workers=num_workers,worker_init_fn=seed_worker,
            sampler=val_sampler)
    )

def get_fgvc_aircraft_loaders(
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    FGVC Aircraft dataset loaders with standard augmentation
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
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
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    train_dataset = datasets.FGVCAircraft(
        './data',
        split='train',
        download=True,
        transform=train_transform
    )
    
    val_dataset = datasets.FGVCAircraft(
        './data',
        split='test',  # FGVCAircraft uses 'test' instead of 'val'
        download=True,
        transform=val_transform
    )
    
    set_seed(seed)
    train_sampler = DeterministicSampler(len(train_dataset))
    val_sampler = DeterministicSampler(len(val_dataset))


    return (
        DataLoader(train_dataset, batch_size=batch_size,  num_workers=num_workers, worker_init_fn=seed_worker,
            sampler=train_sampler),
        DataLoader(val_dataset, batch_size=batch_size,  num_workers=num_workers,worker_init_fn=seed_worker,
            sampler=val_sampler)
    )



def get_flowers102_loaders(
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Oxford Flowers-102 dataset loaders with standard augmentation
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
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
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    train_dataset = datasets.Flowers102(
        './data',
        split='train',
        download=True,
        transform=train_transform
    )
    
    val_dataset = datasets.Flowers102(
        './data',
        split='val',
        download=True,
        transform=val_transform
    )
    
    set_seed(seed)
    train_sampler = DeterministicSampler(len(train_dataset))
    val_sampler = DeterministicSampler(len(val_dataset))


    return (
        DataLoader(train_dataset, batch_size=batch_size,  num_workers=num_workers, worker_init_fn=seed_worker,
            sampler=train_sampler),
        DataLoader(val_dataset, batch_size=batch_size,  num_workers=num_workers,worker_init_fn=seed_worker,
            sampler=val_sampler)
    )

def get_inaturalist_mini_loaders(batch_size: int = 32, num_workers: int = 2, seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    """iNaturalist mini-subset dataset loaders with standard augmentation"""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.INaturalist(
        './data',
        version='2021_mini',
        target_type='super',
        download=True,
        transform=train_transform
    )
    
    val_dataset = datasets.INaturalist(
        './data', 
        version='2021_mini',
        target_type='super',
        download=True,
        transform=val_transform
    )

    set_seed(seed)
    train_sampler = DeterministicSampler(len(train_dataset))
    val_sampler = DeterministicSampler(len(val_dataset))
    
    return (
        DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, worker_init_fn=seed_worker, sampler=train_sampler),
        DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, worker_init_fn=seed_worker, sampler=val_sampler)
    )

def get_food101_loaders(batch_size: int = 32, num_workers: int = 2, seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    """Food-101 dataset loaders with standard augmentation"""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.Food101(
        './data',
        split='train',
        download=True,
        transform=train_transform
    )
    
    val_dataset = datasets.Food101(
        './data',
        split='test',
        download=True,
        transform=val_transform
    )

    set_seed(seed)
    train_sampler = DeterministicSampler(len(train_dataset))
    val_sampler = DeterministicSampler(len(val_dataset))
    
    return (
        DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, worker_init_fn=seed_worker, sampler=train_sampler),
        DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, worker_init_fn=seed_worker, sampler=val_sampler)
    )


dataset_configs = {
        "mnist": {
            "loader": get_mnist_loaders,
            "input_size": 28*28,
            "output_size": 10,
            "weight": 1.0
        },
        "cifar10": {
            "loader": get_cifar10_loaders,
            "input_size": 32*32*3,
            "output_size": 10,
            "weight": 1.0
        },
        "cifar100": {
            "loader": get_cifar100_loaders,
            "input_size": 32*32*3,
            "output_size": 100,
            "weight": 1.0
        },
        "imagenet": {
            "loader": get_imagenet_1k_loaders,
            "input_size": (3, 224, 224),
            "output_size": 1000,
            "weight": 2.0,
            "learning_rate": 0.1
        },
        "shakespeare": {
            "loader": get_shakespeare_loaders,
            "input_size": 32,
            "output_size": None,  # Will be set after loading
            "hidden_size": 32,
            "learning_rate": 3e-4,
            "weight": 10.0
        },
        "flowers102": {
            "loader": get_flowers102_loaders,
            "input_size": (3, 224, 224),
            "output_size": 102,  # Flowers102 has 102 classes
            "weight": 1.5,
            "learning_rate": 0.01
        },
        "fgvc_aircraft": {
            "loader": get_fgvc_aircraft_loaders,
            "input_size": (3, 224, 224),
            "output_size": 100,  # FGVC Aircraft has 100 classes
            "weight": 1.5,
            "learning_rate": 0.01
        },
        "imagenette": {
            "loader": get_imagenette_loaders,
            "input_size": (3, 224, 224),
            "output_size": 10,  # Imagenette has 10 classes
            "weight": 1.0,
            "learning_rate": 0.01
        },
        "inaturalist_mini": {
            "loader": get_inaturalist_mini_loaders,
            "input_size": (3, 224, 224),
            "output_size": 10,
            "weight": 1.0
        },
        "food101": {
            "loader": get_food101_loaders, 
            "input_size": (3, 224, 224),
            "output_size": 101,
            "weight": 1.0
        }
    }

def load_datasets(dataset_names: Union[str, List[str]], batch_size: int = 32, seed=42) -> List[DatasetSpec]:
    """
    Load specified datasets based on input names.
    
    Args:
        dataset_names: Single dataset name or list of dataset names
        batch_size: Batch size for data loaders
    
    Returns:
        List of DatasetSpec objects for requested datasets
    """
    # Convert single string to list for consistent processing
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    
    # Dictionary mapping dataset names to their loader functions and specs
    

    
    dataset_specs = []
    
    for name in dataset_names:
        if name not in dataset_configs:
            raise ValueError(f"Dataset '{name}' not recognized. Available datasets: {list(dataset_configs.keys())}")
            
        config = dataset_configs[name]
        
        if name == "shakespeare":
            train_loader, val_loader, vocab_size = config["loader"](batch_size=batch_size, seed=seed)
            config["output_size"] = vocab_size
        else:
            train_loader, val_loader = config["loader"](batch_size=batch_size, seed=seed)
        
        spec = DatasetSpec(
            name=name,
            input_size=config["input_size"],
            output_size=config["output_size"],
            train_loader=train_loader,
            val_loader=val_loader,
            hidden_size=config.get("hidden_size", 128),
            learning_rate=config.get("learning_rate", 0.001),
            weight=config.get("weight", 1.0)
        )
        
        dataset_specs.append(spec)
    
    return dataset_specs