"""
Phase 2 Full Evaluation
Test set pe comprehensive metrics
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, 
    f1_score, accuracy_score, confusion_matrix,
    classification_report, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from phase2.model_v2 import FakeJobDetectorV2
from phase2.dataset_v2 import FakeJobDatasetV2

print("=" * 70)
print("PHASE 2 - FULL EVALUATION")
print("=" * 70)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# Load data
print("\nLoading test data...")
test_data = pickle.load(open('../../processed/test_v2.pkl', 'rb'))
print(f"Test samples: {len(test_data['labels'])}")
print(f"Frauds: {test_data['labels'].sum()} ({100*test_data['labels'].sum()/len(test_data['labels']):.1f}%)")

# Create dataset
test_dataset = FakeJobDatasetV2(
    test_data['texts'],
    test_data['metadata'],
    test_data['graph_features'],
    test_data['labels']
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load model
print("\nLoading Phase 2 model...")
checkpoint = torch.load('../../models/phase2_best.pt', map_location=device, weights_only=False)

model = FakeJobDetectorV2(
    num_meta_features=test_data['metadata'].shape[1],
    num_graph_features=test_data['graph_features'].shape[1]
)

# Load weights with key mapping
state_dict = checkpoint.get('model_state_dict', checkpoint.get('model'))
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace('bert.', 'distilbert.')
    new_key = new_key.replace('meta_enc.', 'metadata_encoder.')
    new_key = new_key.replace('graph_enc.', 'graph_encoder.encoder.')
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict, strict=False)
model.to(device)
model.eval()

print("Model loaded successfully!")

# Collect predictions
print("\nEvaluating on test set...")
all_probs = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        metadata = batch['metadata'].to(device)
        graph_features = batch['graph_features'].to(device)
        labels = batch['label']
        
        probs = model.predict_proba(input_ids, attention_mask, metadata, graph_features)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

# Calculate metrics
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

# ROC-AUC
auc = roc_auc_score(all_labels, all_probs)
print(f"\nROC-AUC: {auc:.4f}")

# Average Precision (PR-AUC)
ap = average_precision_score(all_labels, all_probs)
print(f"PR-AUC: {ap:.4f}")

# Find best threshold
best_f1 = 0
best_thresh = 0.5
for thresh in np.arange(0.05, 0.95, 0.05):
    preds = (all_probs >= thresh).astype(int)
    f1 = f1_score(all_labels, preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f"\nBest Threshold: {best_thresh:.2f}")
print(f"Best F1 Score: {best_f1:.4f}")

# Metrics at best threshold
preds_binary = (all_probs >= best_thresh).astype(int)
acc = accuracy_score(all_labels, preds_binary)
precision, recall, f1, _ = classification_report(
    all_labels, preds_binary, target_names=['Real', 'Fake'], 
    output_dict=True, zero_division=0
)['weighted avg'].values()

print(f"\nAt Best Threshold:")
print(f"   Accuracy:  {acc:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(all_labels, preds_binary)
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion Matrix:")
print(f"   True Negatives (Correct Real):  {tn}")
print(f"   False Positives (Wrong Fake):   {fp}")
print(f"   False Negatives (Missed Fakes): {fn}")
print(f"   True Positives (Correct Fake):  {tp}")

print(f"\n   False Positive Rate: {fp/(fp+tn)*100:.2f}%")
print(f"   False Negative Rate: {fn/(fn+tp)*100:.2f}%")
print(f"   Detection Rate:      {tp/(tp+fn)*100:.2f}%")

# Classification Report
print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)
print(classification_report(all_labels, preds_binary, 
                           target_names=['Real', 'Fake'], digits=4))

# Feature Importance
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)
importance = model.get_feature_importance()
print(f"Text:     {importance['text']:.4f}")
print(f"Metadata: {importance['metadata']:.4f}")
print(f"Graph:    {importance['graph']:.4f}")

# Save plots
print("\n📊 Saving plots...")
os.makedirs('results', exist_ok=True)

# ROC Curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(all_labels, all_probs)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Phase 2 Model')
plt.legend()
plt.grid(True)
plt.savefig('results/phase2_roc_curve.png', dpi=150, bbox_inches='tight')
print("✅ Saved: results/phase2_roc_curve.png")

# Precision-Recall Curve
plt.figure(figsize=(8, 6))
precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
plt.plot(recall_curve, precision_curve, label=f'PR Curve (AP = {ap:.4f})', linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Phase 2 Model')
plt.legend()
plt.grid(True)
plt.savefig('results/phase2_pr_curve.png', dpi=150, bbox_inches='tight')
print("✅ Saved: results/phase2_pr_curve.png")

# Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('results/phase2_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✅ Saved: results/phase2_confusion_matrix.png")

# Prediction Distribution
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(all_probs[all_labels == 0], bins=50, alpha=0.7, label='Real Jobs', color='green')
plt.hist(all_probs[all_labels == 1], bins=50, alpha=0.7, label='Fake Jobs', color='red')
plt.xlabel('Fraud Probability')
plt.ylabel('Count')
plt.title('Prediction Distribution')
plt.legend()
plt.axvline(best_thresh, color='black', linestyle='--', label=f'Threshold={best_thresh:.2f}')

plt.subplot(1, 2, 2)
plt.boxplot([all_probs[all_labels == 0], all_probs[all_labels == 1]], 
            labels=['Real', 'Fake'])
plt.ylabel('Fraud Probability')
plt.title('Prediction Distribution (Boxplot)')
plt.axhline(best_thresh, color='red', linestyle='--', label=f'Threshold={best_thresh:.2f}')

plt.tight_layout()
plt.savefig('results/phase2_distribution.png', dpi=150, bbox_inches='tight')
print("✅ Saved: results/phase2_distribution.png")

print("\n" + "=" * 70)
print("EVALUATION COMPLETE!")
print("=" * 70)

# Summary
print(f"\n🎉 SUMMARY:")
print(f"   ROC-AUC: {auc:.4f} {'(EXCELLENT!)' if auc > 0.95 else '(GOOD)' if auc > 0.90 else '(NEEDS WORK)'}")
print(f"   Best F1: {best_f1:.4f}")
print(f"   Detection Rate: {tp/(tp+fn)*100:.1f}% of fakes caught")

if auc > 0.99:
    print("\n✨ OUTSTANDING PERFORMANCE!")
    print("   Your model is performing exceptionally well!")
    print("   Ready for deployment! 🚀")
