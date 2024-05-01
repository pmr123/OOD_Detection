import torch
from torch import nn
from torchvision import datasets, transforms

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
            nn.Sigmoid()  # Ensure output range [0, 1] if input is normalized
        )
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        flat_x = x.view(x.size(0), -1)  # Flatten the image
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

def train_vae(model, data_loader, optimizer, epochs=10, device=torch.device('cpu')):
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
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(data_loader.dataset)}')

# Parameters and setup
input_dim = 3 * 32 * 32  
latent_dim = 50
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE(input_dim, latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

from dataloader import get_dataloader_vae
train_loader, _ = get_dataloader_vae('cifar10', batch_size=128)

train_vae(model, train_loader, optimizer, epochs=10, device=device)
# file to code functions specific to method 2
