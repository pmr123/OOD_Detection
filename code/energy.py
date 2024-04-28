# import libraries
import torch
import numpy as np
from utils import extract_feature_maps

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class for energy-like ood detection
class ELOOD:
    def __init__(self, model, T=100):
        self.model = model.to(device)
        self.T = T
    
    
    # reduce array from [batch, n, m, m] to [batch, n] using norm
    # if array is already [batch, n] then return without change
    def reduce_feats(self,feats):
        if len(feats.shape) == 2:
            return feats
        
        output = np.zeros(shape=(feats.shape[0], feats.shape[1]))
        
        # get norms
        for i in range(feats.shape[0]):
            for j in range(feats.shape[1]):
                output[i][j] = np.linalg.norm(feats[i][j])
    
        return output
    
    # get energy function values
    # input size = [batch, n]
    # output size = [batch]
    def energy_vals(self, feats):
        # Perform element-wise division by T
        feats = np.divide(feats, self.T)      

        # Get the exponential of all values in the tensor
        feats = np.exp(feats)

        # Compute row-wise sum
        feats = np.sum(feats, axis=1)

        # Perform element-wise multiplication by -T
        feats = np.multiply(feats, (self.T))

        return feats


    # get ood scores 
    def get_scores(self, loader):
        scores = []
        for i, (images, _) in enumerate(loader):
            images = images.to(device)
            
            # get feature maps
            feats = extract_feature_maps(self.model, images)

            # get latent space representation
            _, _, _, z = self.model(images)

            # append z to feats
            feats.append(z.detach().cpu().data.numpy())

            # reduce dims
            feats = [self.reduce_feats(elem) for elem in feats]

            # get energy function scores
            feats = [self.energy_vals(elem) for elem in feats]

            # compute sum
            total = np.zeros_like(feats[0])
            for elem in feats:
                total = np.add(total, elem)

            # get average
            total = np.divide(total, len(feats))   

            # add scores to list
            scores.extend(total)

        return scores
    
