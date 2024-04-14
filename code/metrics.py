import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix

class Metrics:
    def __init__(self, model, data_looker, device):
        self.model = model
        self.data_looker = data_looker
        self.device = device

    def evaluate(self):
        self.model.eval()
        labels = []
        scores = []
        
        with torch.no_grad():
            for data, target in self.data_looker.load_data():
                data = data.to(self.device)
                recon, mu, logvar = self.model(data)
                recon_error = torch.mean((data - recon) ** 2, dim=[1, 2, 3])
                scores.extend(recon_error.cpu().numpy())
                labels.extend(target.cpu().numpy())

        return np.array(labels), np.array(scores)

    def calculate_metrics(self, labels, scores, threshold):
        predictions = (scores > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        tpr = tp / (tp + fn)
        npv = tn / (tn + fn)
        return {
            "TPR": tpr,  # True Positive Rate
            "NPV": npv   # Negative Predictive Value
        }

    def threshold_sweep(self, labels, scores, num_thresholds=100):
        thresholds = np.linspace(scores.min(), scores.max(), num=num_thresholds)
        tpr_values = []
        npv_values = []

        for threshold in thresholds:
            metrics = self.calculate_metrics(labels, scores, threshold)
            tpr_values.append(metrics["TPR"])
            npv_values.append(metrics["NPV"])

        return thresholds, tpr_values, npv_values

    def plot_metrics(self, thresholds, tpr_values, npv_values):
        plt.figure(figsize=(12, 6))
        plt.plot(thresholds, tpr_values, label='True Positive Rate')
        plt.plot(thresholds, npv_values, label='Negative Predictive Value')
        plt.xlabel('Threshold')
        plt.ylabel('Value')
        plt.title('TPR and NPV vs. Threshold')
        plt.legend()
        plt.grid(True)
        plt.show()

# Example Usage
# Assuming `model`, `data_looker`, `device` are already defined:
metrics = Metrics(model, data_looker, device)
labels, scores = metrics.evaluate()
thresholds, tpr_values, npv_values = metrics.threshold_sweep(labels, scores)
metrics.plot_metrics(thresholds, tpr_values, npv_values)
