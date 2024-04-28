# file to code model architectures

import torch
import torch.nn as nn

# code VAE model architecture

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# VAE model
class VAE(nn.Module):
    def __init__(self, latent_size=100):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 2 * self.latent_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, 128 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu, logvar = torch.split(encoded, self.latent_size, dim=1)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        return decoded, mu, logvar, z

# get trained model
def get_trained_model(modelname):
    model = VAE()
    name = r'../models/'+str(modelname)+'.pt'
    model.load_state_dict(torch.load(name))
    return model


