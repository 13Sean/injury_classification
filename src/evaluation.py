
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import label_binarize


def calculate_metrics(y_true, y_pred, num_classes=5):

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    confusion = confusion_matrix(y_true, y_pred)


    sensitivities = []
    specificities = []
    for i in range(num_classes):
        TP = confusion[i, i]
        FN = np.sum(confusion[i, :]) - TP
        TN = np.sum(confusion) - np.sum(confusion[i, :]) - np.sum(confusion[:, i]) + TP
        FP = np.sum(confusion[:, i]) - TP
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        sensitivities.append(sensitivity)
        specificities.append(specificity)

    mean_sensitivity = np.mean(sensitivities)
    mean_specificity = np.mean(specificities)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': confusion,
        'mean_sensitivity': mean_sensitivity,
        'mean_specificity': mean_specificity,
        'sensitivities': sensitivities,
        'specificities': specificities
    }


def calculate_roc_curves(y_true, y_pred, num_classes=5):

    y_test_bin = label_binarize(y_true, classes=range(num_classes))
    y_pred_bin = label_binarize(y_pred, classes=range(num_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_bin.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    return fpr, tpr, roc_auc


def plot_loss_curve(train_losses, valid_losses, num_epochs, save_path='loss_curve.png'):

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()


def plot_micro_roc(fpr, tpr, roc_auc, save_path='micro_roc.png'):

    plt.figure(figsize=(10, 6))
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()


def plot_class_roc(fpr, tpr, roc_auc, class_names, num_classes=5, save_path='class_roc.png'):

    plt.figure(figsize=(10, 6))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i],
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Class-wise ROC Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()


def print_results(metrics, roc_auc, feature_dim, class_names):

    print('\n\033[indicators: Temperature, Systolic Pressure, Diastolic Pressure, Heart Rate, Respiratory Rate, BSL\033[0m')
    print(f'\033[feature_dim: {feature_dim} (包含FFT特征)\033[0m')
    print(f'\033[mean_sensitivity: {metrics["mean_sensitivity"]:.4f}\033[0m')
    print(f'\033[mean_specificity: {metrics["mean_specificity"]:.4f}\033[0m')
    print(f'\033[AUC: {roc_auc["micro"]:.4f}\033[0m')
    print(f'\033[accuracy: {metrics["accuracy"]:.4f}\033[0m')
    print(f'\033[f1: {metrics["f1"]:.4f}\033[0m')
    print(f'\033[precision: {metrics["precision"]:.4f}\033[0m')
    print(f'\033[recall: {metrics["recall"]:.4f}\033[0m')
    print(f'\033[confusion_matrix:\n{metrics["confusion_matrix"]}\033[0m')
