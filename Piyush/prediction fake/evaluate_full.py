"""
Step 2: Full Evaluation on Test Set
Phase 2 Model (DistilBERT + Metadata + Graph)
Metrics: AUC, F1, Precision, Recall, Confusion Matrix
"""
import sys
sys.path.append('src')

import torch
import pickle
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, accuracy_score, confusion_matrix,
    classification_report, precision_recall_fscore_support
)
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from phase2.model_v2 import FakeJobDetectorV2

print("=" * 70)
print("  STEP 2: FULL EVALUATION ON TEST SET")
print("  Phase 2 Model (DistilBERT + Metadata + Graph Features)")
print("=" * 70)

# ===================================================================
# 1. Load Model
# ===================================================================
print("\n[1/5] Loading model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

checkpoint = torch.load('models/phase2_best.pt', map_location=device, weights_only=False)

with open('processed/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

model = FakeJobDetectorV2(num_meta_features=len(feature_names), num_graph_features=14)

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

print(f"  Model loaded! (epoch {checkpoint.get('epoch', '?')})")
print(f"  Features: {len(feature_names)} metadata + 14 graph = {len(feature_names)+14} total")

# ===================================================================
# 2. Load Test Data
# ===================================================================
print("\n[2/5] Loading test data...")
with open('processed/test_v2.pkl', 'rb') as f:
    test_data = pickle.load(f)

texts = test_data['texts']
metadata = test_data['metadata']
graph_features = test_data['graph_features']
labels = test_data['labels']

print(f"  Samples: {len(labels)}")
print(f"  Fraudulent: {int(sum(labels))} ({100*sum(labels)/len(labels):.1f}%)")
print(f"  Legitimate: {int(len(labels)-sum(labels))} ({100*(1-sum(labels)/len(labels)):.1f}%)")

# ===================================================================
# 3. Run Predictions (batched)
# ===================================================================
print("\n[3/5] Running predictions on test set...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

BATCH_SIZE = 32
all_probs = []

# Tokenize all texts first
print("  Tokenizing texts...")
encodings = tokenizer(
    texts, max_length=512, padding='max_length',
    truncation=True, return_tensors='pt'
)

# Create dataset
meta_tensor = torch.tensor(np.array(metadata), dtype=torch.float32)
graph_tensor = torch.tensor(np.array(graph_features), dtype=torch.float32)
input_ids = encodings['input_ids']
attention_mask = encodings['attention_mask']

dataset = TensorDataset(input_ids, attention_mask, meta_tensor, graph_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

print("  Running model inference...")
with torch.no_grad():
    for batch in tqdm(dataloader, desc="  Evaluating"):
        b_input_ids, b_attention_mask, b_meta, b_graph = batch
        b_input_ids = b_input_ids.to(device)
        b_attention_mask = b_attention_mask.to(device)
        b_meta = b_meta.to(device)
        b_graph = b_graph.to(device)

        probs = model.predict_proba(b_input_ids, b_attention_mask, b_meta, b_graph)
        all_probs.extend(probs.cpu().numpy())

all_probs = np.array(all_probs)
all_labels = np.array(labels)

# ===================================================================
# 4. Calculate Metrics
# ===================================================================
print("\n[4/5] Calculating metrics...")

# AUC
auc = roc_auc_score(all_labels, all_probs)

# Find optimal threshold (maximize F1)
best_f1 = 0
best_thresh = 0.5
for thresh in np.arange(0.01, 0.95, 0.01):
    y_pred = (all_probs >= thresh).astype(int)
    f1 = f1_score(all_labels, y_pred, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

# Predictions at optimal threshold
y_pred_opt = (all_probs >= best_thresh).astype(int)
acc_opt = accuracy_score(all_labels, y_pred_opt)
prec, rec, f1_val, _ = precision_recall_fscore_support(
    all_labels, y_pred_opt, average='binary', zero_division=0
)

# Predictions at threshold 0.5
y_pred_50 = (all_probs >= 0.5).astype(int)
acc_50 = accuracy_score(all_labels, y_pred_50)
f1_50 = f1_score(all_labels, y_pred_50, zero_division=0)

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(all_labels, y_pred_opt).ravel()

# Uncertainty analysis (where LLM would help)
uncertain_mask = (all_probs >= 0.05) & (all_probs <= 0.85)
uncertain_count = uncertain_mask.sum()
uncertain_fraud = all_labels[uncertain_mask].sum()

# ===================================================================
# Print Results
# ===================================================================
print("\n" + "=" * 70)
print("  EVALUATION RESULTS - TEST SET")
print("=" * 70)

print(f"\n  ROC-AUC Score:      {auc:.4f}")
print(f"  Optimal Threshold:  {best_thresh:.2f}")

print(f"\n  --- At Optimal Threshold ({best_thresh:.2f}) ---")
print(f"  Accuracy:           {acc_opt:.4f} ({acc_opt*100:.1f}%)")
print(f"  F1 Score:           {f1_val:.4f}")
print(f"  Precision:          {prec:.4f} ({prec*100:.1f}%)")
print(f"  Recall:             {rec:.4f} ({rec*100:.1f}%)")

print(f"\n  --- At Default Threshold (0.50) ---")
print(f"  Accuracy:           {acc_50:.4f} ({acc_50*100:.1f}%)")
print(f"  F1 Score:           {f1_50:.4f}")

print(f"\n  --- Confusion Matrix (threshold={best_thresh:.2f}) ---")
print(f"  True Negatives  (legit correctly identified):  {tn}")
print(f"  True Positives  (fraud correctly caught):      {tp}")
print(f"  False Positives (legit wrongly flagged):       {fp}")
print(f"  False Negatives (fraud MISSED):                {fn}")

print(f"\n  --- Classification Report ---")
print(classification_report(
    all_labels, y_pred_opt,
    target_names=['Legitimate', 'Fraudulent'],
    digits=4
))

print(f"  --- LLM Enhancement Opportunity ---")
print(f"  Uncertain predictions (0.05-0.85):  {uncertain_count} ({100*uncertain_count/len(all_labels):.1f}%)")
print(f"  Frauds in uncertain zone:           {int(uncertain_fraud)}")
print(f"  These are the cases where Gemini LLM would help!")

# Error analysis
fn_mask = (y_pred_opt == 0) & (all_labels == 1)
fp_mask = (y_pred_opt == 1) & (all_labels == 0)
print(f"\n  --- Error Analysis ---")
print(f"  Missed frauds (False Negatives):    {fn}")
if fn > 0:
    fn_probs = all_probs[fn_mask]
    print(f"    Avg probability of missed frauds: {fn_probs.mean():.4f}")
    print(f"    Min: {fn_probs.min():.4f}, Max: {fn_probs.max():.4f}")
print(f"  False alarms (False Positives):     {fp}")
if fp > 0:
    fp_probs = all_probs[fp_mask]
    print(f"    Avg probability of false alarms:  {fp_probs.mean():.4f}")

# ===================================================================
# 5. Save Plots
# ===================================================================
print("\n[5/5] Saving plots...")
os.makedirs('models/eval_plots', exist_ok=True)

# ROC Curve
fpr, tpr, _ = roc_curve(all_labels, all_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Phase 2 Model (AUC = {auc:.4f})', linewidth=2, color='#2196F3')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Baseline')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Phase 2 Model', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('models/eval_plots/roc_curve.png', dpi=150)
plt.close()
print("  Saved: models/eval_plots/roc_curve.png")

# Precision-Recall Curve
precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, linewidth=2, color='#4CAF50')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('models/eval_plots/pr_curve.png', dpi=150)
plt.close()
print("  Saved: models/eval_plots/pr_curve.png")

# Confusion Matrix
cm = confusion_matrix(all_labels, y_pred_opt)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Fraudulent'],
            yticklabels=['Legitimate', 'Fraudulent'],
            annot_kws={"size": 14})
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title(f'Confusion Matrix (threshold={best_thresh:.2f})', fontsize=14)
plt.tight_layout()
plt.savefig('models/eval_plots/confusion_matrix.png', dpi=150)
plt.close()
print("  Saved: models/eval_plots/confusion_matrix.png")

# Probability Distribution
plt.figure(figsize=(10, 5))
plt.hist(all_probs[all_labels == 0], bins=50, alpha=0.6, label='Legitimate', color='#4CAF50')
plt.hist(all_probs[all_labels == 1], bins=50, alpha=0.6, label='Fraudulent', color='#F44336')
plt.axvline(x=best_thresh, color='black', linestyle='--', label=f'Threshold ({best_thresh:.2f})')
plt.xlabel('Predicted Fraud Probability', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Prediction Distribution by Class', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('models/eval_plots/prediction_distribution.png', dpi=150)
plt.close()
print("  Saved: models/eval_plots/prediction_distribution.png")

print("\n" + "=" * 70)
print("  EVALUATION COMPLETE!")
print("=" * 70)
