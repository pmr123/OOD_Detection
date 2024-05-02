# import libraries
import torch
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class for energy-like ood detection
class ELOOD:
    def __init__(self, model, T=10000):
        self.model = model.to(device)
        self.thresholds_min = 0
        self.thresholds_max = 1
        
    # get reconstruction term of energy per sample
    def get_recon_energy(self, output, input):
        result = torch.abs(output - input)
        result = torch.exp(result)
        result = torch.sum(result)
        return result.item()

    # get ood scores 
    def get_scores(self, loader):
        ood_scores = []
        for i, (images, _) in enumerate(loader):
            images = images.to(device)
            
            # get reconstructed, mu, logvar, latent space representation
            reconstructed, mu, logvar, z = self.model(images)


            # get reconstruction term of energy
            recon_energy = torch.zeros((images.shape[0]))
            for i in range(images.shape[0]):
                recon_energy[i] = self.get_recon_energy(reconstructed[i], images[i])
            
            kld = torch.zeros((images.shape[0]))
            for i in range(images.shape[0]):
                kld[i] = - 0.5 * (1 + logvar[i] - mu[i].pow(2) - logvar[i].exp()).sum()

            scores = recon_energy.cpu().data.numpy() + kld.cpu().data.numpy()
            
            ood_scores.extend(scores) 
        
        return ood_scores
    

    # function to train thresholds for ood scores
    def train_ood(self, train_dl):
        scores = self.get_scores(train_dl)
        quartiles = np.quantile(scores, [0.25, 0.75])
        self.thresholds_min = quartiles[0]
        self.thresholds_max = quartiles[1]
    
    # predict if samples are in distribution or out distribution
    def predict_ood(self, test_dl):
        scores = self.get_scores(test_dl)
        preds = np.zeros_like(scores)
        for i in scores:
            if scores[i] >= self.thresholds_min and scores[i] <= self.thresholds_max:
                preds[i] = 1
            else:
                preds[i] = 0

        return preds 
    
