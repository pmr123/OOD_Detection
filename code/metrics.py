import torch
import numpy as np
import matplotlib.pyplot as plt
from models import VAE  # Import the VAE class from models.py
from dataloader import get_dataloader_vae  # Assuming the data loader is set up correctly
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latent_size = 100  # Adjust this to match your model configuration
model = VAE(latent_size).to(device)
data_loader, _ = get_dataloader_vae('cifar10', batch_size=128)

class Metrics:
    def __init__(self, model, data_loader, device):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device
        self.labels = []
        self.scores = []

    def evaluate(self):
        self.model.eval()
        self.labels = []
        self.scores = []
        with torch.no_grad():
            for data, target in self.data_loader:
                data = data.to(self.device)
                outputs, _, _, _ = self.model(data) 
                outputs = outputs.view(outputs.shape[0], -1)

                self.scores.append(outputs.cpu().numpy())
                self.labels.append(target.cpu().numpy())

        self.labels = np.concatenate(self.labels)
        self.scores = np.concatenate(self.scores, axis=0)
        return self.labels, self.scores

    def calculate_metrics_at_fpr95(self):
        classes = np.unique(self.labels)
        y_bin = label_binarize(self.labels, classes=classes)
        roc_aucs = []
        for i in range(y_bin.shape[1]):
            if np.sum(y_bin[:, i]) > 0:  # Only calculate if there are positive samples
                current_scores = self.scores[:, i].ravel()  # Flatten the array to 1D if not already
                fpr, tpr, thresholds = roc_curve(y_bin[:, i], current_scores)
                roc_auc = auc(fpr, tpr)
                roc_aucs.append(roc_auc)
                idx = np.min(np.where(fpr <= 0.95))  # Find first index where FPR is below or equal to 0.95
                print(f"Class {classes[i]}: TPR at FPR 95%: {tpr[idx]} with Threshold: {thresholds[idx]}")
        mean_auc = np.mean(roc_aucs) if roc_aucs else 0
        return {"Mean AUROC": mean_auc}

    def plot_metrics(self, thresholds, fpr_values, tpr_values):
        plt.figure(figsize=(12, 6))
        plt.plot(thresholds, tpr_values, label='True Positive Rate')
        plt.plot(thresholds, fpr_values, label='False Positive Rate')
        plt.xlabel('Threshold')
        plt.ylabel('Rate')
        plt.title('TPR and FPR vs. Threshold')
        plt.legend()
        plt.grid(True)
        plt.show()

    def run_metrics_evaluation(self):
        labels, scores = self.evaluate()
        metrics = self.calculate_metrics_at_fpr95()
        print("Mean AUROC:", metrics["Mean AUROC"])

        classes = np.unique(self.labels)
        y_bin = label_binarize(self.labels, classes=classes)
        fpr_values = []
        tpr_values = []
        all_thresholds = []  # Changed variable name to all_thresholds
        for i in range(y_bin.shape[1]):
            if np.sum(y_bin[:, i]) > 0:  # Only calculate if there are positive samples
                current_scores = self.scores[:, i].ravel() 
                fpr, tpr, thresholds = roc_curve(y_bin[:, i], current_scores)
                fpr_values.append(fpr)
                tpr_values.append(tpr)
                all_thresholds.append(thresholds)  
        
        self.plot_metrics(all_thresholds[0], fpr_values[0], tpr_values[0])  



metrics = Metrics(model, data_loader, device)
metrics.run_metrics_evaluation()
