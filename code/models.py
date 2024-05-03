# file to code model architectures

import torch
import torch.nn as nn

# code VAE model architecture

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# VAE model
class VAE(nn.Module):
    def __init__(self, latent_size=256):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size= 3, stride = 2, padding  = 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size= 3, stride = 2, padding  = 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size= 3, stride = 2, padding  = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size= 3, stride = 2, padding  = 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size= 3, stride = 2, padding  = 1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size= 3, stride = 2, padding  = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * self.latent_size),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(2 * self.latent_size, self.latent_size)
        self.fc_var = nn.Linear(2 * self.latent_size, self.latent_size)

        self.decoder_input = nn.Sequential(
            nn.Linear(self.latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512 * 4),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride = 2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride = 2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride = 2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride = 2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride = 2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size= 3, stride= 2, padding  = 1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size= 3, stride= 2, padding  = 1),
            nn.Tanh()
        )

    def encode(self, x):
        x = self.encoder(x)
        # Split the result into mu and logvar components
        # of the latent Gaussian distribution
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return mu, log_var
    
    def decode(self, z):
        z = self.decoder_input(z)
        z = z.view(-1, 512, 2, 2)
        z = self.decoder(z)
        return z


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z)
        return decoded, mu, logvar, z

# get trained model
def get_trained_model(modelname):
    model = VAE()
    name = r'../models/'+str(modelname)+'.pt'
    model.load_state_dict(torch.load(name))
    return model


