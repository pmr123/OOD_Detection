import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix

class Metrics:
    def __init__(self, model, data_looker, device):
        self.model = model.to(device)
        self.data_looker = data_looker
        self.device = device
        self.labels = []  # Storing labels as a class attribute
        self.scores = []  # Storing scores as a class attribute

    def evaluate(self):
        self.model.eval()
        self.labels = []  # Reset labels for each and every evaluation
        self.scores = []  # Reset scores for each and every evaluation
        with torch.no_grad():
            for data, target in self.data_looker.load_data():
                data = data.to(self.device)
                ood_scores, _ = self.model(data)  # Extract only the first output
                self.scores.extend(ood_scores.cpu().numpy())
                self.labels.extend(target.cpu().numpy())
        return np.array(self.labels), np.array(self.scores)

    def calculate_metrics_at_fpr95(self):
        fpr, tpr, thresholds = roc_curve(self.labels, self.scores) #scores because it gives the output that indicates the likelihood or the confidence in prediction
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(self.labels, self.scores)
        pr_auc = average_precision_score(self.labels, self.scores)

        idx = np.argmin(np.abs(fpr - 0.95))
        tpr_at_fpr95 = tpr[idx]
        return {
            "AUROC": roc_auc,
            "AUPR": pr_auc,
            "TPR_at_FPR95": tpr_at_fpr95,
            "Threshold_at_FPR95": thresholds[idx]
        }

    def threshold_sweep(self, num_thresholds=100):
        thresholds = np.linspace(min(self.scores), max(self.scores), num_thresholds)
        tpr_values = []
        fpr_values = []

        for threshold in thresholds:
            predictions = (self.scores >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(self.labels, predictions).ravel()
            tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
            tpr_values.append(tpr)
            fpr_values.append(fpr)

        return thresholds, fpr_values, tpr_values

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
        self.evaluate()
        metrics = self.calculate_metrics_at_fpr95()
        print("AUROC:", metrics["AUROC"], "AUPR:", metrics["AUPR"], "TPR at FPR 95%:", metrics["TPR_at_FPR95"], "with Threshold:", metrics["Threshold_at_FPR95"])
        thresholds, fpr_values, tpr_values = self.threshold_sweep()
        self.plot_metrics(thresholds, fpr_values, tpr_values)

# Example Usage
# Assuming that model, data_looker, device have already been defined
metrics = Metrics(model, data_looker, device)
metrics.run_metrics_evaluation()
