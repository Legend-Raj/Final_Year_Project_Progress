"""
Test Phase 2 Model - With Graph Features
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import pickle
import numpy as np
from transformers import DistilBertTokenizer

from phase2.model_v2 import FakeJobDetectorV2
from phase2.network_features import NetworkFeatureExtractor

print("=" * 60)
print("PHASE 2 MODEL TESTING")
print("=" * 60)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Load checkpoint
checkpoint = torch.load('models/phase2_best.pt', map_location=device, weights_only=False)
print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
print(f"Validation AUC: {checkpoint.get('auc', checkpoint.get('best_val_auc', 'N/A'))}")

# Create model
model = FakeJobDetectorV2(num_meta_features=41, num_graph_features=14)
# Load state dict with key mapping (Colab names -> Local names)
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

# Load scaler and feature names
scaler = pickle.load(open('processed/scaler.pkl', 'rb'))
feature_names = pickle.load(open('processed/feature_names.pkl', 'rb'))

# Initialize tokenizer and network extractor
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
net_extractor = NetworkFeatureExtractor()

print("\n" + "=" * 60)
print("TEST CASES")
print("=" * 60)

def predict_job(job_dict):
    """Predict on a job posting"""
    # Prepare text
    text = f"Title: {job_dict.get('title', '')}. "
    text += f"Description: {job_dict.get('description', '')}. "
    text += f"Requirements: {job_dict.get('requirements', '')}. "
    text += f"Company: {job_dict.get('company_profile', '')}. "
    text += f"Benefits: {job_dict.get('benefits', '')}"
    
    # Tokenize
    encoding = tokenizer(text, max_length=512, padding='max_length', 
                        truncation=True, return_tensors='pt')
    
    # Metadata features
    features = {}
    features['desc_length'] = len(str(job_dict.get('description', '')))
    features['title_length'] = len(str(job_dict.get('title', '')))
    features['req_length'] = len(str(job_dict.get('requirements', '')))
    features['company_profile_length'] = len(str(job_dict.get('company_profile', '')))
    features['has_salary'] = 1 if job_dict.get('salary_range') else 0
    features['has_department'] = 1 if job_dict.get('department') else 0
    features['has_company_logo'] = int(job_dict.get('has_company_logo', 0))
    features['has_questions'] = int(job_dict.get('has_questions', 0))
    features['telecommuting'] = int(job_dict.get('telecommuting', 0))
    features['has_location'] = 1 if job_dict.get('location') else 0
    features['is_us_location'] = 1 if 'US' in str(job_dict.get('location', '')) else 0
    
    exp_mapping = {'Not Applicable': 0, 'Entry level': 1, 'Associate': 2,
                   'Mid-Senior level': 3, 'Director': 4, 'Executive': 5}
    features['experience_level'] = exp_mapping.get(job_dict.get('required_experience'), 0)
    
    edu_mapping = {'Unspecified': 0, 'High School or equivalent': 1, 'Vocational': 2,
                   'Some College Coursework Completed': 3, 'Associate Degree': 4,
                   "Bachelor's Degree": 5, "Master's Degree": 6, 'Doctorate': 7, 'Professional': 8}
    features['education_level'] = edu_mapping.get(job_dict.get('required_education'), 0)
    
    industry = job_dict.get('industry', '')
    for fname in feature_names:
        if fname.startswith('industry_'):
            features[fname] = 1 if industry == fname.replace('industry_', '') else 0
    
    function = job_dict.get('function', '')
    for fname in feature_names:
        if fname.startswith('function_'):
            features[fname] = 1 if function == fname.replace('function_', '') else 0
    
    features['desc_missing'] = 1 if not job_dict.get('description') else 0
    features['requirements_missing'] = 1 if not job_dict.get('requirements') else 0
    
    feature_vector = [features.get(f, 0) for f in feature_names]
    meta_scaled = scaler.transform([feature_vector])[0]
    
    # Graph features (PHASE 2 - NEW!)
    graph_obj = net_extractor.extract(
        company_profile=job_dict.get('company_profile', ''),
        contact_email=job_dict.get('contact_email'),
        company_name=job_dict.get('company_name', job_dict.get('title', ''))
    )
    graph_vector = graph_obj.to_vector()
    
    # Convert to tensors
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    meta_tensor = torch.tensor([meta_scaled], dtype=torch.float32).to(device)
    graph_tensor = torch.tensor([graph_vector], dtype=torch.float32).to(device)
    
    # Predict
    with torch.no_grad():
        prob = model.predict_proba(input_ids, attention_mask, meta_tensor, graph_tensor)
    
    prob_value = prob.item()
    
    # Risk level
    threshold = 0.10  # Best threshold from training
    if prob_value < threshold * 0.5:
        risk = "LOW"
    elif prob_value < threshold * 1.5:
        risk = "MEDIUM"
    else:
        risk = "HIGH"
    
    return {
        'probability': prob_value,
        'is_fake': prob_value > threshold,
        'risk': risk,
        'graph_features': {
            'domain_trust_score': graph_obj.domain_trust_score,
            'email_is_free': graph_obj.email_is_free_provider,
            'has_linkedin': graph_obj.has_linkedin_page,
            'is_suspicious_domain': graph_obj.is_known_fake_domain
        }
    }

# Test 1: Legitimate Job
test_jobs = [
    {
        'name': 'Legitimate Software Engineer',
        'data': {
            'title': 'Senior Software Engineer',
            'description': 'Join our engineering team at Google. We are looking for experienced developers.',
            'requirements': '5+ years Python, CS degree',
            'company_profile': 'Google LLC is an American multinational technology company. Founded in 1998.',
            'benefits': 'Health insurance, 401k, competitive salary',
            'salary_range': '$150k-$200k',
            'department': 'Engineering',
            'has_company_logo': 1,
            'has_questions': 1,
            'telecommuting': 1,
            'location': 'US, CA, Mountain View',
            'employment_type': 'Full-time',
            'required_experience': 'Mid-Senior level',
            'required_education': "Bachelor's Degree",
            'industry': 'Computer Software',
            'function': 'Engineering',
            'contact_email': 'careers@google.com'
        }
    },
    {
        'name': 'Suspicious Work-From-Home',
        'data': {
            'title': 'Work From Home - $500/Day Guaranteed!',
            'description': 'No experience needed! Send $50 for starter kit. Earn easy money from home!',
            'requirements': 'No requirements. Anyone can apply.',
            'company_profile': 'Quick money opportunities for everyone! Contact us now.',
            'benefits': 'Unlimited earning potential! Work 1 hour per day!',
            'salary_range': '$500/day',
            'department': None,
            'has_company_logo': 0,
            'has_questions': 0,
            'telecommuting': 1,
            'location': None,
            'employment_type': 'Full-time',
            'required_experience': 'Not Applicable',
            'required_education': 'Unspecified',
            'industry': None,
            'function': None,
            'contact_email': 'quickmoney@gmail.com'
        }
    },
    {
        'name': 'Marketing Intern (Legit)',
        'data': {
            'title': 'Marketing Intern',
            'description': 'Looking for a marketing intern to help with social media campaigns.',
            'requirements': 'Currently enrolled in college, marketing major preferred',
            'company_profile': 'TechStartup Inc. We are a growing tech company founded in 2018.',
            'benefits': 'Flexible hours, learning opportunities',
            'salary_range': None,
            'department': 'Marketing',
            'has_company_logo': 1,
            'has_questions': 0,
            'telecommuting': 0,
            'location': 'US, NY, New York',
            'employment_type': 'Full-time',
            'required_experience': 'Entry level',
            'required_education': "Bachelor's Degree",
            'industry': 'Marketing and Advertising',
            'function': 'Marketing',
            'contact_email': 'jobs@techstartup.com'
        }
    }
]

for test in test_jobs:
    print(f"\n--- {test['name']} ---")
    result = predict_job(test['data'])
    
    print(f"Fraud Probability: {result['probability']:.2%}")
    print(f"Risk Level: {result['risk']}")
    print(f"Is Fake: {result['is_fake']}")
    print("\nGraph Features Used:")
    for key, value in result['graph_features'].items():
        print(f"  {key}: {value}")

print("\n" + "=" * 60)
print("TESTING COMPLETE!")
print("=" * 60)
