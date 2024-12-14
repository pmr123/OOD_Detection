# import libraries
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set random seed
np.random.seed(42)

# class for energy-like ood detection
class ELOOD:
    def __init__(self, model):
        # model
        self.model = model.to(device)

        # range threshold
        self.thresholds_min = 0
        self.thresholds_max = 1

        # train ood scores
        self.train_scores = []
        
    # get reconstruction term of energy per sample
    def get_recon_energy(self, output, input):
        # Energy = sum( e^(|x[i] - f(x[i])|) ) 
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
            
            # get penalty term of energy (KL divergence)
            kld = torch.zeros((images.shape[0]))
            for i in range(images.shape[0]):
                kld[i] = - 0.5 * (1 + logvar[i] - mu[i].pow(2) - logvar[i].exp()).sum()

            # add to get total energy
            scores = recon_energy.cpu().data.numpy() + kld.cpu().data.numpy()
            
            ood_scores.extend(scores) 
        
        return ood_scores
    

    # function to train thresholds for ood scores
    def train_ood(self, train_dl):
        # get scores
        scores = self.get_scores(train_dl)
        # update self variables
        self.train_scores = scores
        quartiles = np.quantile(scores, [0.25, 0.75])
        self.thresholds_min = quartiles[0]
        self.thresholds_max = quartiles[1]
    
    # predict if samples are in distribution or out distribution
    def predict_ood(self, test_dl, already_scores = False):
        # get scores

        if already_scores:
            scores = test_dl
        else:
            scores = self.get_scores(test_dl)

        # predict based on if scores are within range or not
        preds = np.zeros_like(scores)

        for i in range(len(scores)):
            if self.thresholds_min <= scores[i] and scores[i] <= self.thresholds_max:
                preds[i] = 0
            else:
                preds[i] = 1

        return preds 
    
    # utility function to get metrics
    def get_metrics(self, preds):

        # choose preds.shape[0] random scores from in distribution samples
        id_scores = np.random.choice(self.train_scores, size=preds.shape[0])

        # get predictions on these scores
        id_preds = self.predict_ood(id_scores, already_scores=True)

        # get true labels
        tl1 = np.zeros_like(preds)
        tl2 = np.ones_like(preds)

        # concatenate to get true labels and predictions
        true_labels = np.append(tl1, tl2)
        predictions = np.append(id_preds, preds)

        # get metrics

        # False Positive Rate when True Positive Rate = 95%
        fpr, tpr, thresholds = roc_curve(true_labels, predictions)
        false_positive_rate_95 = fpr[np.argmax(tpr >= 0.95)]

        # AUPR (Area Under the Precision-Recall Curve)
        precision, recall, _ = precision_recall_curve(true_labels, predictions)
        aupr = auc(recall, precision)

        # AUROC (Area Under the Receiver Operating Characteristic Curve)
        fpr, tpr, _ = roc_curve(true_labels, predictions)
        auroc = auc(fpr, tpr)

        # plot the roc curve for the model
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auroc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

        return(false_positive_rate_95, aupr, auroc)
    
