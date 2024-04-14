# file to model utility functions
import torch
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                      
# model train loop function
def train_vae(model, optimizer, num_epochs, train_loader, test_loader, modelname):
    train_loss = []
    test_loss = []
    num_train = 0
    num_test = 0
    best_loss = np.inf

    for epoch in range(num_epochs):
        # train mode
        model.train()
        avg_train_loss = 0
        for i, (images, _) in enumerate(train_loader):
            num_train += images.shape[0]
            images = images.to(device)

            # Forward pass
            recon_images, mu, logvar = model(images)
            loss = 0.5 * (recon_images - images).pow(2).sum() - 0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
            avg_train_loss += loss

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_train_loss /= num_train          
        
        
        # eval mode
        model.eval()
        avg_val_loss = 0
        for i, (images, _) in enumerate(test_loader):
            num_test += images.shape[0]
            images = images.to(device)

            # Forward pass
            with torch.no_grad():
                recon_images, mu, logvar = model(images)
                loss = 0.5 * (recon_images - images).pow(2).sum() - 0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
                avg_val_loss += loss
        
        avg_val_loss /= num_test

        # logging losses
        train_loss.append(avg_train_loss)
        test_loss.append(avg_train_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss.item():.4f}, Validation Loss: {avg_val_loss.item():.4f}')

        if avg_val_loss.item() < best_loss:
            print(f'Best Validation Loss reduced from {best_loss:.4f} to {avg_val_loss.item():.4f}')
            print("Saving Model")
            best_loss = avg_val_loss.item()
            name = r'../models/'+str(modelname)+'.pt'
            torch.save(model.state_dict(), name)

    return model, train_loss, test_loss

# data loader

# data transformers