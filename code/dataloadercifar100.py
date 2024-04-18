import torch
from torchvision import datasets, transforms

def load_cifar100_data():
    batch_size = 128
    normalize = transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255, 66.7/255))

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    transform_test = transforms.Compose([
        transforms.CenterCrop(size=(32, 32)),
        transforms.ToTensor(),
        normalize
    ])

    cifar100_train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('data', train=True, download=True, transform=transform_train),
        batch_size=batch_size, shuffle=True)

    cifar100_test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('data', train=False, download=True, transform=transform_test),
        batch_size=batch_size)

    return cifar100_train_loader, cifar100_test_loader

def load_cifar10_data():
    
    normalize = transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255, 66.7/255))

    transform_test = transforms.Compose([
        transforms.CenterCrop(size=(32, 32)),
        transforms.ToTensor(),
        normalize
    ])

    cifar10_test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=False, download=True, transform=transform_test),
        batch_size=1, shuffle=False)  

    return cifar10_test_loader

def load_svhn_data():
    
    normalize = transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255, 66.7/255))

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    svhn_test_loader = torch.utils.data.DataLoader(
        datasets.SVHN('data', split="test", download=True, transform=transform_test),
        batch_size=1, shuffle=True)

    return svhn_test_loader

def load_lsun_data():
    normalize = transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255, 66.7/255))
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        normalize
    ])

    lsun_test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('LSUN/', transform=transform_test),
        batch_size=1, shuffle=True)

    return lsun_test_loader