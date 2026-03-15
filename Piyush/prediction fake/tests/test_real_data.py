"""
Test model on actual data from dataset
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pickle
import numpy as np
from src.predict import FakeJobPredictor

print("=" * 60)
print("TESTING ON REAL DATASET SAMPLES")
print("=" * 60)

# Load test data
test_data = pickle.load(open('processed/test.pkl', 'rb'))

# Find some fake and real jobs
fake_indices = np.where(test_data['labels'] == 1)[0]
real_indices = np.where(test_data['labels'] == 0)[0]

print(f"\nTotal test samples: {len(test_data['labels'])}")
print(f"Fake jobs: {len(fake_indices)}")
print(f"Real jobs: {len(real_indices)}")

# Initialize predictor
predictor = FakeJobPredictor()

# Test on 3 FAKE jobs
print("\n" + "=" * 60)
print("TESTING ON ACTUAL FAKE JOBS FROM DATASET")
print("=" * 60)

for i, idx in enumerate(fake_indices[:3]):
    print(f"\n--- FAKE JOB {i+1} ---")
    text = test_data['texts'][idx]
    meta = test_data['metadata'][idx]
    
    # Show first 200 chars of text
    print(f"Text preview: {text[:200]}...")
    
    # Need to create a job_dict from the metadata and text
    # For now, let's just use the raw prediction
    from transformers import DistilBertTokenizer
    import torch
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    encoding = tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    device = predictor.device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    meta_tensor = torch.tensor([meta], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        prob = predictor.model.predict_proba(input_ids, attention_mask, meta_tensor)
    
    prob_value = prob.item()
    print(f"Fraud Probability: {prob_value:.4f} ({prob_value*100:.1f}%)")
    
    if prob_value < 0.05:
        risk = "LOW"
    elif prob_value < 0.15:
        risk = "MEDIUM"
    else:
        risk = "HIGH"
    
    print(f"Risk Level: {risk}")
    print(f"ACTUAL LABEL: FAKE")

# Test on 3 REAL jobs
print("\n" + "=" * 60)
print("TESTING ON ACTUAL REAL JOBS FROM DATASET")
print("=" * 60)

for i, idx in enumerate(real_indices[:3]):
    print(f"\n--- REAL JOB {i+1} ---")
    text = test_data['texts'][idx]
    meta = test_data['metadata'][idx]
    
    print(f"Text preview: {text[:200]}...")
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    encoding = tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    device = predictor.device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    meta_tensor = torch.tensor([meta], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        prob = predictor.model.predict_proba(input_ids, attention_mask, meta_tensor)
    
    prob_value = prob.item()
    print(f"Fraud Probability: {prob_value:.4f} ({prob_value*100:.1f}%)")
    
    if prob_value < 0.05:
        risk = "LOW"
    elif prob_value < 0.15:
        risk = "MEDIUM"
    else:
        risk = "HIGH"
    
    print(f"Risk Level: {risk}")
    print(f"ACTUAL LABEL: REAL")

print("\n" + "=" * 60)
print("Testing complete!")
print("=" * 60)
