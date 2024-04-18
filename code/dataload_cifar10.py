import os
import numpy as np
import torch
from torchvision import datasets, transforms

def get_dataloader_cifar10(dataset_name, train=True, batch_size=128):
    # Normalisation
    mean = np.array([[125.3 / 255, 123.0 / 255, 113.9 / 255]]).T
    std = np.array([[63.0 / 255, 62.1 / 255.0, 66.7 / 255.0]]).T
    normalize = transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0))

    # transformations
    transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
])

    # Selecting the correct dataset and transform
    if dataset_name.lower() == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=True, download=True, transform=transform_train),
            batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=False, transform=transform_test),
            batch_size=batch_size)
        train_data = list(torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=True, download=True, transform=transform_test),
            batch_size=1, shuffle=False))

        test_data = list(torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=False, download=True, transform=transform_test),
            batch_size=1, shuffle=False))
    
    #out of order distribution
    elif dataset_name.lower() == 'cifar100':
        dataset = datasets.CIFAR100('data', train=False, download=True, transform=transform_test)
    elif dataset_name.lower() == 'svhn':
        dataset = datasets.SVHN('data', split="test", download=True, transform=transform_test)
    elif dataset_name.lower() == 'lsun':
        dataset_path = 'LSUN/'
        dataset = datasets.ImageFolder(dataset_path, transform=transform_test)
    else:
        raise ValueError("Unsupported dataset")

    # DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return dataloader


