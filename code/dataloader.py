import numpy as np
import torch
from torchvision import datasets, transforms

def get_dataloader_vae(dataset_name, train=True, batch_size=128):
    # Normalisation
    normalize = transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255, 66.7 / 255))

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
        dataset_class = datasets.CIFAR10
    elif dataset_name.lower() == 'cifar100':
        dataset_class = datasets.CIFAR100
    else:
        raise ValueError("Unsupported dataset")

    train_loader = torch.utils.data.DataLoader(
        dataset_class('data', train=True, download=True, transform=transform_train),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset_class('data', train=False, transform=transform_test),
        batch_size=batch_size)

    return train_loader, test_loader  # Returning both loaders

def get_dataloader_OOD(ood_dataset_name, batch_size=128):
    # Normalisation
    normalize = transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255, 66.7 / 255))
    resize = transforms.Resize(size=(32,32)) # resizing to cifar10/100 size
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        resize,
        normalize
    ])
    

    # OOD Dataset
    if ood_dataset_name.lower() == 'cifar10':
        dataset = datasets.CIFAR10('data', train=False, download=True, transform=transform_test)
    elif ood_dataset_name.lower() == 'cifar100':
        dataset = datasets.CIFAR100('data', train=False, download=True, transform=transform_test)
    elif ood_dataset_name.lower() == 'svhn':
        dataset = datasets.SVHN('data', split="test", download=True, transform=transform_test)
    elif ood_dataset_name.lower() == 'lsun': 
        dataset_path = 'LSUN/'
        dataset = datasets.ImageFolder(dataset_path, transform=transform_test)
    else:
        return ValueError("Unsupported OOD dataset: {ood_dataset_name}")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)  
    return dataloader
