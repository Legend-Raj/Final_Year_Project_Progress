"""
Evaluation script - comprehensive model evaluation
"""
import torch
import pickle
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, 
    f1_score, accuracy_score, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from dataset import FakeJobDataset
from model import FakeJobDetector


def plot_roc_curve(y_true, y_scores, save_path='models/roc_curve.png'):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"ROC curve saved to {save_path}")
    plt.close()


def plot_precision_recall_curve(y_true, y_scores, save_path='models/pr_curve.png'):
    """Plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig(save_path)
    print(f"PR curve saved to {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path='models/confusion_matrix.png'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def find_optimal_threshold(y_true, y_scores):
    """Find threshold that maximizes F1 score"""
    best_f1 = 0
    best_thresh = 0.5
    
    for thresh in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_scores >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    return best_thresh, best_f1


def evaluate_model(model_path='../models/best_model_full.pt', split='test'):
    """Comprehensive model evaluation"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading {split} data...")
    data = pickle.load(open(f'../processed/{split}.pkl', 'rb'))
    
    dataset = FakeJobDataset(
        data['texts'],
        data['metadata'],
        data['labels']
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Load model
    print("Loading model...")
    feature_names = pickle.load(open('../processed/feature_names.pkl', 'rb'))
    num_meta_features = len(feature_names)
    
    model = FakeJobDetector(num_meta_features=num_meta_features)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Collect predictions
    all_probs = []
    all_labels = []
    
    print("Running evaluation...")
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            metadata = batch['metadata'].to(device)
            labels = batch['label']
            
            probs = model.predict_proba(input_ids, attention_mask, metadata)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Metrics
    auc = roc_auc_score(all_labels, all_probs)
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS ({split.upper()} SET)")
    print(f"{'='*50}")
    print(f"ROC-AUC: {auc:.4f}")
    
    # Find optimal threshold
    opt_thresh, opt_f1 = find_optimal_threshold(all_labels, all_probs)
    print(f"Optimal Threshold: {opt_thresh:.2f} (F1: {opt_f1:.4f})")
    
    # Default threshold (0.5)
    y_pred_05 = (all_probs >= 0.5).astype(int)
    acc_05 = accuracy_score(all_labels, y_pred_05)
    f1_05 = f1_score(all_labels, y_pred_05, zero_division=0)
    
    print(f"\nAt threshold 0.5:")
    print(f"  Accuracy: {acc_05:.4f}")
    print(f"  F1 Score: {f1_05:.4f}")
    
    # At optimal threshold
    y_pred_opt = (all_probs >= opt_thresh).astype(int)
    acc_opt = accuracy_score(all_labels, y_pred_opt)
    
    print(f"\nAt optimal threshold {opt_thresh:.2f}:")
    print(f"  Accuracy: {acc_opt:.4f}")
    print(f"  F1 Score: {opt_f1:.4f}")
    
    # Classification report
    print(f"\nClassification Report (at threshold {opt_thresh:.2f}):")
    print(classification_report(all_labels, y_pred_opt, 
                                target_names=['Real', 'Fake'],
                                digits=4))
    
    # Confusion matrix values
    tn, fp, fn, tp = confusion_matrix(all_labels, y_pred_opt).ravel()
    print(f"Confusion Matrix:")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives:  {tp}")
    
    # Save plots
    os.makedirs('models', exist_ok=True)
    plot_roc_curve(all_labels, all_probs)
    plot_precision_recall_curve(all_labels, all_probs)
    plot_confusion_matrix(all_labels, y_pred_opt)
    
    # Error analysis - look at high confidence errors
    print(f"\n{'='*50}")
    print("ERROR ANALYSIS")
    print(f"{'='*50}")
    
    # False Positives (predicted fake, actually real)
    fp_mask = (y_pred_opt == 1) & (all_labels == 0)
    fp_high_conf = all_probs[fp_mask] > 0.8
    print(f"False Positives with >80% confidence: {fp_high_conf.sum()}")
    
    # False Negatives (predicted real, actually fake)
    fn_mask = (y_pred_opt == 0) & (all_labels == 1)
    fn_high_conf = all_probs[fn_mask] < 0.2
    print(f"False Negatives with <20% confidence: {fn_high_conf.sum()}")
    
    return {
        'auc': auc,
        'optimal_threshold': opt_thresh,
        'f1': opt_f1,
        'accuracy': acc_opt,
        'predictions': all_probs,
        'labels': all_labels
    }


if __name__ == "__main__":
    # Evaluate on test set
    results = evaluate_model(split='test')
    
    # Also evaluate on validation set for comparison
    print("\n" + "="*50)
    val_results = evaluate_model(split='val')
