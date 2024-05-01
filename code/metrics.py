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

        # Compute ROC curve and ROC area for each class
        metrics_dict = {}
        for i, class_ in enumerate(classes):
            current_scores = scores[:, i]
            fpr, tpr, thresholds = roc_curve(y_bin[:, i], current_scores)
            roc_auc = auc(fpr, tpr)
            precision, recall, pr_thresholds = precision_recall_curve(y_bin[:, i], current_scores)
            pr_auc = average_precision_score(y_bin[:, i], current_scores)
            idx = np.min(np.where(fpr <= 0.95)) if np.any(fpr <= 0.95) else -1
            metrics_dict[class_] = {
                "AUROC": roc_auc,
                "AUPR": pr_auc,
                "TPR_at_FPR95": tpr[idx] if idx != -1 else None,
                "Threshold_at_FPR95": thresholds[idx] if idx != -1 else None
            }

        # Print each class's metrics
        for key, value in metrics_dict.items():
            print(f"Class {key}: AUROC: {value['AUROC']:.2f}, AUPR: {value['AUPR']:.2f}, "
                  f"TPR at FPR 95%: {value['TPR_at_FPR95']}, with Threshold: {value['Threshold_at_FPR95']}")

        # Plot all ROC curves
        plt.figure()
        colors = iter(['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'black'])
        for i, class_ in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], scores[:, i])
            plt.plot(fpr, tpr, color=next(colors), lw=2, label=f'Class {class_} ROC curve (area = {metrics_dict[class_]["AUROC"]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for each class')
        plt.legend(loc="lower right")
        plt.show()

metrics = Metrics(model, data_loader, device)
metrics.plot_roc_curves()

