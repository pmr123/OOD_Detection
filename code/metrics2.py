import torch
import numpy as np
import matplotlib.pyplot as plt
from models import VAE
from dataloader import get_dataloader_vae
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latent_size = 100
model = VAE(latent_size).to(device)
data_loader, _ = get_dataloader_vae('cifar10', batch_size=128)

class Metrics:
    def __init__(self, model, data_loader, device):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device

    def evaluate(self):
        self.model.eval()
        labels, scores = [], []
        with torch.no_grad():
            for data, target in self.data_loader:
                data = data.to(self.device)
                outputs, _, _, _ = self.model(data)
                outputs = outputs.view(outputs.shape[0], -1)
                scores.append(outputs.cpu().numpy())
                labels.append(target.cpu().numpy())
        labels = np.concatenate(labels)
        scores = np.concatenate(scores, axis=0)
        return labels, scores

    def plot_roc_curves(self):
        labels, scores = self.evaluate()
        classes = np.unique(labels)
        y_bin = label_binarize(labels, classes=classes)

        metrics_dict = {}
        plt.figure()
        colors = iter(['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'black'])
        for i, class_ in enumerate(classes):
            current_scores = scores[:, i]
            fpr, tpr, thresholds = roc_curve(y_bin[:, i], current_scores)
            roc_auc = auc(fpr, tpr)
            precision, recall, pr_thresholds = precision_recall_curve(y_bin[:, i], current_scores)
            pr_auc = average_precision_score(y_bin[:, i], current_scores)
            
            idx_tpr95 = np.where(tpr >= 0.95)[0][0] if np.any(tpr >= 0.95) else -1
            metrics_dict[class_] = {
                "AUROC": roc_auc,
                "AUPR": pr_auc,
                "FPR_at_TPR95": fpr[idx_tpr95] if idx_tpr95 != -1 else None,
                "Threshold_at_TPR95": thresholds[idx_tpr95] if idx_tpr95 != -1 else None
            }
            print(f'Class {class_}: AUROC: {roc_auc:.2f}, AUPR: {pr_auc:.2f}, '
                  f'FPR at TPR 95%: {metrics_dict[class_]["FPR_at_TPR95"]}, '
                  f'with Threshold: {metrics_dict[class_]["Threshold_at_TPR95"]}')

         


metrics = Metrics(model, data_loader, device)
metrics.plot_roc_curves()