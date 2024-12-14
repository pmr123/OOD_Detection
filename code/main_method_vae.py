import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import torch
from torchvision import datasets, transforms
import tarfile
import os
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from metrics2 import *

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load SVHN dataset
svhn = list(torch.utils.data.DataLoader(
    datasets.SVHN('data', split="test", download=True,
                   transform=transform_test),
    batch_size=1, shuffle=True))

lsun_tar_path = 'LSUN.tar.gz'
lsun_extract_folder = 'LSUN/'

# Extract the LSUN dataset
with tarfile.open(lsun_tar_path, "r:gz") as tar:
    tar.extractall(path=lsun_extract_folder)

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the LSUN dataset
lsun_dataset = datasets.ImageFolder(root=lsun_extract_folder, transform=transform_test)
lsun_loader = torch.utils.data.DataLoader(lsun_dataset, batch_size=1, shuffle=True)

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim*2)  # Outputs mean and log variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        flat_x = x.view(x.size(0), -1)
        params = self.encoder(flat_x)
        mu, log_var = params[:, :latent_dim], params[:, latent_dim:]
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_var

def vae_loss(recon_x, x, mu, log_var):
    x_flat = x.view(x.size(0), -1)
    recon_loss = nn.functional.mse_loss(recon_x, x_flat, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_div

def train_vae(model, data_loader, optimizer, epochs=1, device=torch.device('cpu')):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, _ in data_loader:
            data = data.to(device)
            optimizer.zero_grad()
            reconstruction, mu, log_var = model(data)
            loss = vae_loss(reconstruction, data, mu, log_var)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}')

# Parameters and setup
input_dim = 3 * 32 * 32
latent_dim = 50
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CIFAR datasets
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
])

cifar10_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
cifar100_dataset = datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)

cifar10_loader = DataLoader(cifar10_dataset, batch_size=128, shuffle=True)
cifar100_loader = DataLoader(cifar100_dataset, batch_size=128, shuffle=True)

model_cifar10 = VAE(input_dim, latent_dim).to(device)
model_cifar100 = VAE(input_dim, latent_dim).to(device)

optimizer_cifar10 = torch.optim.Adam(model_cifar10.parameters(), lr=learning_rate)
optimizer_cifar100 = torch.optim.Adam(model_cifar100.parameters(), lr=learning_rate)

print("Training on CIFAR-10:")
train_vae(model_cifar10, cifar10_loader, optimizer_cifar10, epochs=1, device=device)

print("Training on CIFAR-100:")
train_vae(model_cifar100, cifar100_loader, optimizer_cifar100, epochs=1, device=device)

# Define the path to the LSUN tar.gz file and the directory to extract to
lsun_tar_path = 'LSUN.tar.gz'
lsun_extract_folder = 'LSUN/'

# Extract the LSUN dataset
with tarfile.open(lsun_tar_path, "r:gz") as tar:
    tar.extractall(path=lsun_extract_folder)

# Load LSUN dataset
lsun_dataset = datasets.ImageFolder(root=lsun_extract_folder, transform=transform)
lsun_loader = DataLoader(lsun_dataset, batch_size=128, shuffle=True)

# Load SVHN dataset
svhn_dataset = datasets.SVHN(root='./data', split='train', transform=transforms.ToTensor(), download=True)
svhn_loader = DataLoader(svhn_dataset, batch_size=128, shuffle=True)

# Test the models on LSUN and SVHN
print("Testing CIFAR-10 model on LSUN:")
train_vae(model_cifar10, lsun_loader, optimizer_cifar10, epochs=1, device=device)

print("Testing CIFAR-10 model on SVHN:")
train_vae(model_cifar10, svhn_loader, optimizer_cifar10, epochs=1, device=device)

print("Testing CIFAR-100 model on LSUN:")
train_vae(model_cifar100, lsun_loader, optimizer_cifar100, epochs=1, device=device)

print("Testing CIFAR-100 model on SVHN:")
train_vae(model_cifar100, svhn_loader, optimizer_cifar100, epochs=1, device=device)

class Metrics:
    def __init__(self, model, data_loader, device):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device

    def evaluate(self):
        self.model.eval()
        recon_errors, true_labels = [], []
        with torch.no_grad():
            for data, _ in self.data_loader:
                data = data.to(self.device)
                reconstruction, _, _ = self.model(data)
                reconstruction = reconstruction.view(data.size(0), -1)
                data = data.view(data.size(0), -1)
                mse = ((reconstruction - data) ** 2).mean(dim=1).cpu().numpy()
                recon_errors.extend(mse)
                threshold = np.percentile(mse, 95)  
                labels = (mse > threshold).astype(int)
                true_labels.extend(labels)

        return np.array(true_labels), np.array(recon_errors)

    def plot_roc_curve(self):
        labels, scores = self.evaluate()
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def plot_precision_recall_curve(self):
        labels, scores = self.evaluate()
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)

        plt.figure()
        plt.step(recall, precision, where='post', label=f'AP={ap:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall curve')
        plt.legend(loc="upper right")
        plt.show()

metrics = Metrics(model_cifar10, cifar10_loader, device) 
metrics.plot_roc_curve()
metrics.plot_precision_recall_curve()


