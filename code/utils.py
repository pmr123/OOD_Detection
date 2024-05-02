# file to model utility functions
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                      
def vae_loss(x, x_recon, mu, logvar):
    #print(x.shape, x_recon.shape)
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div

# model train loop function
def train_vae(model, optimizer, num_epochs, train_loader, test_loader, modelname):
    # send model to device
    model = model.to(device)

    train_loss = []
    test_loss = []
    best_loss = np.inf

    for epoch in range(num_epochs):
        # train mode
        model.train()
        avg_train_loss = 0
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)

            optimizer.zero_grad()

            # Forward pass
            recon_images, mu, logvar, _ = model(images)
            loss = vae_loss(images, recon_images, mu, logvar)
            avg_train_loss += loss.item()

            # Backward and optimize
            loss.backward()
            optimizer.step()
        
        avg_train_loss = avg_train_loss / len(train_loader.dataset)
                 
        # eval mode
        model.eval()
        avg_val_loss = 0
        for i, (images, _) in enumerate(test_loader):
            images = images.to(device)

            # Forward pass
            with torch.no_grad():
                recon_images, mu, logvar, _ = model(images)
                loss = vae_loss(images, recon_images, mu, logvar)
                avg_val_loss += loss.item()

        avg_val_loss = avg_val_loss / len(test_loader.dataset)

        # logging losses
        train_loss.append(avg_train_loss)
        test_loss.append(avg_val_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_loss:
            print(f'Best Validation Loss reduced from {best_loss:.4f} to {avg_val_loss:.4f}')
            print("Saving Model")
            best_loss = avg_val_loss
            name = r'../models/'+str(modelname)+'.pt'
            torch.save(model.state_dict(), name)

    return model, train_loss, test_loss

# Extract feature maps for model
# Note this gets all Conv2d, ConvTranspose2d, and Linear layers but does not get the latent space representation
def extract_feature_maps(model, input_tensor):
    # eval mode
    model.eval()
    feature_maps = []

    def hook(module, input, output):
        feats = output.detach().cpu().data.numpy()
        feature_maps.append(feats)

    hooks = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Linear):
            hooks.append(layer.register_forward_hook(hook))

    with torch.no_grad():
        model(input_tensor)

    for hook in hooks:
        hook.remove()

    return feature_maps
