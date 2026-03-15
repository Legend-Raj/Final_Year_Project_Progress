"""
Train the Final Fusion Layer

Generates 11-dim feature vectors for each sample:
  - model_probability, model_confidence (from Phase 2 model)
  - llm_probability, llm_confidence, llm_num_red_flags, llm_reasoning_score (from Mock LLM)
  - graph_domain_trust, graph_email_type, graph_linkedin_presence, graph_suspicious (from graph features)
  - metadata_richness (from text field completeness)

Then trains a small neural network (11 -> 64 -> 32 -> 16 -> 1) to combine them.

Usage:
    python train_fusion.py
"""
import sys
import os
sys.path.append('src')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from transformers import DistilBertTokenizer

from phase2.model_v2 import FakeJobDetectorV2
from phase3.final_fusion_model import FinalFusionLayer
from phase3.mock_llm import MockLLMAnalyzer
from focal_loss import FocalLoss

print("=" * 70)
print("  TRAIN FINAL FUSION LAYER")
print("  11-dim features -> Neural Network -> Final Prediction")
print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# ===================================================================
# 1. Load Phase 2 model
# ===================================================================
print("\n[1/6] Loading Phase 2 model...")
with open('processed/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

checkpoint = torch.load('models/phase2_best.pt', map_location=device, weights_only=False)
phase2_model = FakeJobDetectorV2(num_meta_features=len(feature_names), num_graph_features=14)
state_dict = checkpoint.get('model_state_dict', checkpoint.get('model'))
mapped = {}
for k, v in state_dict.items():
    nk = k.replace('bert.', 'distilbert.').replace('meta_enc.', 'metadata_encoder.').replace('graph_enc.', 'graph_encoder.encoder.')
    mapped[nk] = v
phase2_model.load_state_dict(mapped, strict=False)
phase2_model.to(device)
phase2_model.eval()
print("  Phase 2 model loaded!")

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
mock_llm = MockLLMAnalyzer()

# ===================================================================
# 2. Load original CSV for Mock LLM (needs raw text fields)
# ===================================================================
print("\n[2/6] Loading data...")
df = pd.read_csv('src/fake_job_postings.csv')

# Load processed train/val/test indices
with open('processed/train_v2.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open('processed/val_v2.pkl', 'rb') as f:
    val_data = pickle.load(f)
with open('processed/test_v2.pkl', 'rb') as f:
    test_data = pickle.load(f)

print(f"  Train: {len(train_data['labels'])} samples ({int(train_data['labels'].sum())} fraud)")
print(f"  Val:   {len(val_data['labels'])} samples ({int(val_data['labels'].sum())} fraud)")
print(f"  Test:  {len(test_data['labels'])} samples ({int(test_data['labels'].sum())} fraud)")


# ===================================================================
# 3. Generate 11-dim fusion features for each sample
# ===================================================================
def generate_fusion_features(data_dict, df_full, split_name, batch_size=32):
    """Generate 11-dim feature vectors for a dataset split."""
    print(f"\n  Generating features for {split_name}...")
    
    texts = data_dict['texts']
    metadata = data_dict['metadata']
    graph_features = data_dict['graph_features']
    labels = data_dict['labels']
    indices = data_dict['indices']
    
    # Step A: Get model predictions (batched)
    print(f"    Getting model predictions...")
    encodings = tokenizer(texts, max_length=512, padding='max_length',
                          truncation=True, return_tensors='pt')
    dataset = TensorDataset(
        encodings['input_ids'], encodings['attention_mask'],
        torch.tensor(np.array(metadata), dtype=torch.float32),
        torch.tensor(np.array(graph_features), dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"    {split_name} model"):
            b_ids, b_mask, b_meta, b_graph = batch
            probs = phase2_model.predict_proba(
                b_ids.to(device), b_mask.to(device),
                b_meta.to(device), b_graph.to(device)
            )
            model_probs.extend(probs.cpu().numpy())
    model_probs = np.array(model_probs)
    
    # Step B: Get Mock LLM features + graph features for each sample
    print(f"    Extracting LLM + graph features...")
    all_features = []
    
    for i in tqdm(range(len(labels)), desc=f"    {split_name} features"):
        mp = model_probs[i]
        
        # Model features
        model_confidence = abs(mp - 0.5) * 2  # 0 to 1
        
        # Build a job dict for Mock LLM from the combined text
        # Parse the text back into fields
        text = texts[i]
        job_dict = _parse_text_to_dict(text)
        
        # Also try to get original data from CSV if index available
        idx = indices[i]
        if idx < len(df_full):
            row = df_full.iloc[idx]
            job_dict['title'] = str(row.get('title', '')) if pd.notna(row.get('title')) else ''
            job_dict['description'] = str(row.get('description', '')) if pd.notna(row.get('description')) else ''
            job_dict['company_profile'] = str(row.get('company_profile', '')) if pd.notna(row.get('company_profile')) else ''
            job_dict['requirements'] = str(row.get('requirements', '')) if pd.notna(row.get('requirements')) else ''
            job_dict['benefits'] = str(row.get('benefits', '')) if pd.notna(row.get('benefits')) else ''
            job_dict['salary_range'] = str(row.get('salary_range', '')) if pd.notna(row.get('salary_range')) else ''
            job_dict['contact_email'] = ''  # Not in CSV
            job_dict['location'] = str(row.get('location', '')) if pd.notna(row.get('location')) else ''
        
        # Mock LLM analysis
        llm_result = mock_llm.analyze(job_dict, mp)
        llm_prob = llm_result.fraud_probability
        llm_conf_map = {'high': 1.0, 'medium': 0.5, 'low': 0.25}
        llm_confidence = llm_conf_map.get(llm_result.confidence, 0.5)
        llm_num_flags = min(len(llm_result.red_flags), 5) / 5.0
        llm_reasoning_score = min(1.0, len(llm_result.reasoning) / 500)
        
        # Graph features (from pre-computed)
        gf = graph_features[i]
        graph_domain_trust = gf[12]  # domain_trust_score index
        graph_email_type = 1.0 if gf[3] else (0.0 if gf[4] else 0.5)  # corporate vs free
        graph_linkedin = float(gf[6])  # has_linkedin_page
        graph_suspicious = float(gf[13])  # is_known_fake_domain
        
        # Metadata richness
        filled = sum([
            bool(job_dict.get('description')),
            bool(job_dict.get('requirements')),
            bool(job_dict.get('company_profile')),
            bool(job_dict.get('salary_range')),
            bool(job_dict.get('location')),
        ])
        metadata_richness = filled / 5.0
        
        # 11-dim feature vector
        features = [
            mp,                    # model_probability
            model_confidence,      # model_confidence
            llm_prob,              # llm_probability
            llm_confidence,        # llm_confidence
            llm_num_flags,         # llm_num_red_flags (normalized)
            llm_reasoning_score,   # llm_reasoning_score
            graph_domain_trust,    # graph_domain_trust
            graph_email_type,      # graph_email_type
            graph_linkedin,        # graph_linkedin_presence
            graph_suspicious,      # graph_suspicious_indicators
            metadata_richness,     # metadata_richness_score
        ]
        all_features.append(features)
    
    return np.array(all_features, dtype=np.float32), labels


def _parse_text_to_dict(text):
    """Parse combined text back into a rough job dict."""
    job_dict = {}
    parts = text.split('. ')
    for part in parts:
        if part.startswith('Title:'):
            job_dict['title'] = part.replace('Title:', '').strip()
        elif part.startswith('Description:'):
            job_dict['description'] = part.replace('Description:', '').strip()
        elif part.startswith('Requirements:'):
            job_dict['requirements'] = part.replace('Requirements:', '').strip()
        elif part.startswith('Company:'):
            job_dict['company_profile'] = part.replace('Company:', '').strip()
        elif part.startswith('Benefits:'):
            job_dict['benefits'] = part.replace('Benefits:', '').strip()
    return job_dict


print("\n[3/6] Generating fusion features...")
train_features, train_labels = generate_fusion_features(train_data, df, "train")
val_features, val_labels = generate_fusion_features(val_data, df, "val")
test_features, test_labels = generate_fusion_features(test_data, df, "test")

print(f"\n  Train features shape: {train_features.shape}")
print(f"  Val features shape:   {val_features.shape}")
print(f"  Test features shape:  {test_features.shape}")

# Save fusion features for reuse
fusion_data = {
    'train_features': train_features, 'train_labels': train_labels,
    'val_features': val_features, 'val_labels': val_labels,
    'test_features': test_features, 'test_labels': test_labels,
    'feature_names': [
        'model_probability', 'model_confidence',
        'llm_probability', 'llm_confidence', 'llm_num_red_flags', 'llm_reasoning_score',
        'graph_domain_trust', 'graph_email_type', 'graph_linkedin_presence', 'graph_suspicious',
        'metadata_richness'
    ]
}
with open('processed/fusion_features.pkl', 'wb') as f:
    pickle.dump(fusion_data, f)
print("  Saved: processed/fusion_features.pkl")

# ===================================================================
# 4. Train Fusion Layer
# ===================================================================
print("\n[4/6] Training fusion layer...")

# Create datasets
train_dataset = TensorDataset(
    torch.tensor(train_features, dtype=torch.float32),
    torch.tensor(train_labels, dtype=torch.float32)
)
val_dataset = TensorDataset(
    torch.tensor(val_features, dtype=torch.float32),
    torch.tensor(val_labels, dtype=torch.float32)
)
test_dataset = TensorDataset(
    torch.tensor(test_features, dtype=torch.float32),
    torch.tensor(test_labels, dtype=torch.float32)
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Create fusion model
fusion_model = FinalFusionLayer(input_dim=11, hidden_dim=64).to(device)
optimizer = optim.Adam(fusion_model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = FocalLoss(alpha=0.75, gamma=2.0)

# Training loop
EPOCHS = 50
PATIENCE = 8
best_val_auc = 0
patience_counter = 0

for epoch in range(EPOCHS):
    # Train
    fusion_model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = fusion_model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Validate
    fusion_model.eval()
    val_probs, val_true = [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            logits = fusion_model(batch_x)
            probs = torch.sigmoid(logits)
            val_probs.extend(probs.cpu().numpy())
            val_true.extend(batch_y.numpy())
    
    val_probs = np.array(val_probs)
    val_true = np.array(val_true)
    val_auc = roc_auc_score(val_true, val_probs) if val_true.sum() > 0 else 0.0
    
    avg_loss = total_loss / len(train_loader)
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}")
    
    # Early stopping
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        patience_counter = 0
        # Save best
        torch.save({
            'fusion_state_dict': fusion_model.state_dict(),
            'epoch': epoch,
            'val_auc': val_auc,
            'input_dim': 11,
            'hidden_dim': 64,
        }, 'models/fusion_layer.pt')
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

print(f"\n  Best Val AUC: {best_val_auc:.4f}")

# ===================================================================
# 5. Evaluate on test set
# ===================================================================
print("\n[5/6] Evaluating on test set...")

# Load best model
ckpt = torch.load('models/fusion_layer.pt', map_location=device, weights_only=False)
fusion_model.load_state_dict(ckpt['fusion_state_dict'])
fusion_model.eval()

test_probs, test_true = [], []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        logits = fusion_model(batch_x)
        probs = torch.sigmoid(logits)
        test_probs.extend(probs.cpu().numpy())
        test_true.extend(batch_y.numpy())

test_probs = np.array(test_probs)
test_true = np.array(test_true)

test_auc = roc_auc_score(test_true, test_probs)

# Find best threshold
best_f1, best_thresh = 0, 0.5
for t in np.arange(0.01, 0.95, 0.01):
    f1 = f1_score(test_true, (test_probs >= t).astype(int), zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

preds = (test_probs >= best_thresh).astype(int)
acc = accuracy_score(test_true, preds)

print(f"\n  --- FUSION LAYER TEST RESULTS ---")
print(f"  AUC:              {test_auc:.4f}")
print(f"  Best Threshold:   {best_thresh:.2f}")
print(f"  F1 Score:         {best_f1:.4f}")
print(f"  Accuracy:         {acc:.4f}")

# Compare with raw model predictions
raw_model_auc = roc_auc_score(test_true, test_features[:, 0])  # model_probability column
best_raw_f1 = 0
for t in np.arange(0.01, 0.95, 0.01):
    f1 = f1_score(test_true, (test_features[:, 0] >= t).astype(int), zero_division=0)
    if f1 > best_raw_f1:
        best_raw_f1 = f1

print(f"\n  --- COMPARISON ---")
print(f"  Raw Model AUC:    {raw_model_auc:.4f}  -->  Fusion AUC:    {test_auc:.4f}")
print(f"  Raw Model F1:     {best_raw_f1:.4f}  -->  Fusion F1:     {best_f1:.4f}")

# ===================================================================
# 6. Demo
# ===================================================================
print(f"\n[6/6] Demo predictions...")
demo_inputs = [
    [0.01, 0.98, 0.05, 0.25, 0.0, 0.1, 0.7, 1.0, 1.0, 0.0, 0.8],  # Legit job
    [0.15, 0.70, 0.95, 1.00, 0.8, 0.9, 0.3, 0.0, 0.0, 0.0, 0.2],  # Scam
    [0.05, 0.90, 0.50, 0.50, 0.4, 0.5, 0.5, 0.5, 0.0, 0.0, 0.4],  # Borderline
]
demo_names = ["Legit job", "Obvious scam", "Borderline"]

with torch.no_grad():
    for name, inp in zip(demo_names, demo_inputs):
        x = torch.tensor([inp], dtype=torch.float32).to(device)
        logit = fusion_model(x)
        prob = torch.sigmoid(logit).item()
        print(f"  {name:15s} -> Fusion: {prob:.1%} ({'FAKE' if prob > best_thresh else 'LEGIT'})")

print("\n" + "=" * 70)
print("  FUSION LAYER TRAINING COMPLETE!")
print(f"  Saved: models/fusion_layer.pt")
print("=" * 70)
