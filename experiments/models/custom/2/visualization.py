import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

def plot_confussion_matrix(y_true, y_pred, class_labels: dict, save_path='results/confussion_matrix.png'):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 2. Confusion Matrix Plot
    plt.figure()
    cm = confusion_matrix(y_true, y_pred)
    # Normalize CM to see percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_labels.keys(), 
                yticklabels=class_labels.keys())
    plt.title('Aggregated Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def report_scores(y_true, y_pred, class_labels, score_reports=None, path="results/scores.txt"):
    # 3. Print Comprehensive Scores
    print("\n" + "="*30)
    print("FINAL 10-FOLD CV RESULTS")
    print("="*30)
    report = classification_report(y_true, y_pred, target_names=list(class_labels.keys()))
    print(report)
    with open(path, 'w') as file:
        file.write(
            "-"*20 + "\n" + 
            report + "\n" +
            f"Final Validation Scores Report:\n \
            Accuracy: {np.mean(score_reports['accuracy'])} \
            F1: {score_reports['f1']} \
            Kappa: {score_reports['kappa']}\n"
        )

def plot_loss_curves(train_losses, val_losses, save_path="results/loss_curves.png" ):
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs, val_losses, marker='o', label='Validation Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()
    # plt.show()
    
def plot_accuracy_curves(train_accs, val_accs, save_path="results/accuracy_curves.png" ):
    epochs = range(1, len(train_accs) + 1)

    plt.figure()
    plt.plot(epochs, train_accs, marker='o', label='Train Acc')
    plt.plot(epochs, val_accs, marker='o', label='Val Acc')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    # plt.show()