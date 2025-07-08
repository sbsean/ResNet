import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split


def get_cifar10_datasets(root='./data', download=True, val_split=0.2):
   
    # Data transformations
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load full training dataset
    full_train_dataset = torchvision.datasets.CIFAR10(
        root=root, 
        train=True, 
        download=download, 
        transform=train_transform
    )
    
    # Load test dataset 
    test_dataset = torchvision.datasets.CIFAR10(
        root=root, 
        train=False, 
        download=download, 
        transform=val_transform
    )
    
    # Split training set into train/val 
    total_size = len(full_train_dataset)
    val_size = int(total_size * val_split)  # 20% for validation
    train_size = total_size - val_size      # 80% for training
    
    train_dataset, val_dataset_from_train = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    

    val_dataset_from_train.dataset = torchvision.datasets.CIFAR10(
        root=root, 
        train=True, 
        download=False, 
        transform=val_transform
    )
    
    return train_dataset, val_dataset_from_train