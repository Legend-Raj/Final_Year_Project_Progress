"""
End-to-End Test: Full Hybrid Pipeline
Phase 2 Model + Gemini LLM working together
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import torch
import pickle
import time
import numpy as np
from transformers import DistilBertTokenizer

from phase2.model_v2 import FakeJobDetectorV2
from phase2.network_features import NetworkFeatureExtractor
from phase3.llm_analyzer_v2 import LLMJobAnalyzer
from phase3.feature_utils import extract_metadata_from_job

print("=" * 70)
print("  FULL HYBRID PIPELINE TEST")
print("  Phase 2 Model + Gemini LLM (end-to-end)")
print("=" * 70)

# ===================================================================
# STEP 1: Load Phase 2 Model
# ===================================================================
print("\n[1/4] Loading Phase 2 model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

checkpoint = torch.load('models/phase2_best.pt', map_location=device, weights_only=False)

with open('processed/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
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
print("  Phase 2 model loaded!")

# ===================================================================
# STEP 2: Initialize helpers
# ===================================================================
print("\n[2/4] Initializing tokenizer and network extractor...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
net_extractor = NetworkFeatureExtractor()
print("  Ready!")

# ===================================================================
# STEP 3: Initialize LLM (Gemini)
# ===================================================================
print("\n[3/4] Initializing LLM analyzer...")
try:
    llm_analyzer = LLMJobAnalyzer(provider="auto")
    llm_available = True
except Exception as e:
    print(f"  LLM not available: {e}")
    print("  Will test model-only pipeline")
    llm_available = False

# ===================================================================
# STEP 4: Define prediction function
# ===================================================================
print("\n[4/4] Pipeline ready!")


def hybrid_predict(job_dict, uncertainty_low=0.25, uncertainty_high=0.75):
    """Full hybrid prediction: Phase 2 model + optional LLM"""
    start_time = time.time()
    
    # --- Phase 2 Model Prediction ---
    text = f"Title: {job_dict.get('title', '')}. "
    text += f"Description: {job_dict.get('description', '')}. "
    text += f"Requirements: {job_dict.get('requirements', '')}. "
    text += f"Company: {job_dict.get('company_profile', '')}. "
    text += f"Benefits: {job_dict.get('benefits', '')}"
    
    encoding = tokenizer(text, max_length=512, padding='max_length',
                         truncation=True, return_tensors='pt')
    
    meta_vector = extract_metadata_from_job(job_dict, feature_names)
    meta_scaled = scaler.transform([meta_vector])[0]
    meta_tensor = torch.tensor([meta_scaled], dtype=torch.float32).to(device)
    
    graph_obj = net_extractor.extract(
        company_profile=job_dict.get('company_profile', ''),
        contact_email=job_dict.get('contact_email'),
        company_name=job_dict.get('company_name', job_dict.get('title', ''))
    )
    graph_tensor = torch.tensor([graph_obj.to_vector()], dtype=torch.float32).to(device)
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        model_prob = model.predict_proba(input_ids, attention_mask,
                                          meta_tensor, graph_tensor).item()
    
    model_time = time.time() - start_time
    
    # --- Build result ---
    result = {
        'model_probability': model_prob,
        'final_probability': model_prob,
        'method': 'model_only',
        'used_llm': False,
        'llm_analysis': None,
        'model_time_ms': round(model_time * 1000),
        'total_time_ms': round(model_time * 1000),
        'graph_features': {
            'domain_trust': graph_obj.domain_trust_score,
            'email_free': bool(graph_obj.email_is_free_provider),
            'has_linkedin': bool(graph_obj.has_linkedin_page),
            'suspicious_domain': bool(graph_obj.is_known_fake_domain),
        }
    }
    
    # --- LLM for uncertain cases ---
    if llm_available and uncertainty_low <= model_prob <= uncertainty_high:
        llm_start = time.time()
        llm_result = llm_analyzer.analyze(job_dict, model_prob)
        llm_time = time.time() - llm_start
        
        # Weighted combination
        if llm_result.confidence == "high":
            final_prob = 0.3 * model_prob + 0.7 * llm_result.fraud_probability
        elif llm_result.confidence == "medium":
            final_prob = 0.5 * model_prob + 0.5 * llm_result.fraud_probability
        else:
            final_prob = 0.7 * model_prob + 0.3 * llm_result.fraud_probability
        
        result['final_probability'] = final_prob
        result['method'] = 'hybrid (model + LLM)'
        result['used_llm'] = True
        result['llm_analysis'] = {
            'provider': llm_result.provider,
            'probability': llm_result.fraud_probability,
            'confidence': llm_result.confidence,
            'red_flags': llm_result.red_flags[:5],
            'reasoning': llm_result.reasoning[:200],
            'cost': llm_result.cost_usd,
            'cached': llm_result.cached,
            'time_ms': round(llm_time * 1000),
        }
        result['total_time_ms'] = round((time.time() - start_time) * 1000)
    
    # Risk level
    prob = result['final_probability']
    if prob < 0.3:
        result['risk_level'] = 'LOW'
    elif prob < 0.7:
        result['risk_level'] = 'MEDIUM'
    else:
        result['risk_level'] = 'HIGH'
    result['is_fake'] = prob > 0.5
    
    return result


# ===================================================================
# TEST CASES
# ===================================================================
test_cases = [
    {
        'name': '1. Obvious Scam (should be HIGH)',
        'data': {
            'title': 'Work From Home - $500/Day Guaranteed!',
            'description': 'No experience needed! Send $50 for starter kit. Earn easy money from home! Contact: quickmoney@gmail.com',
            'requirements': 'No requirements. Anyone can apply.',
            'company_profile': 'Quick money opportunities for everyone! Start earning today!',
            'benefits': 'Unlimited earning potential! Work 1 hour per day!',
            'salary_range': '$500/day',
            'location': None,
            'employment_type': 'Full-time',
            'contact_email': 'quickmoney@gmail.com'
        }
    },
    {
        'name': '2. Legitimate Google Job (should be LOW)',
        'data': {
            'title': 'Senior Software Engineer',
            'description': 'Join our engineering team at Google. We build scalable systems that serve billions of users worldwide. Strong engineering culture with peer reviews.',
            'requirements': '5+ years experience in Python/Java/Go. BS in Computer Science required. Distributed systems experience preferred.',
            'company_profile': 'Google LLC is an American multinational technology company specializing in Internet-related services and products. Founded in 1998 by Larry Page and Sergey Brin.',
            'benefits': 'Competitive salary $150k-$200k, health insurance, 401k matching, free meals, shuttle',
            'salary_range': '$150,000 - $200,000',
            'location': 'Mountain View, CA',
            'employment_type': 'Full-time',
            'has_company_logo': 1,
            'has_questions': 1,
            'department': 'Engineering',
            'required_experience': 'Mid-Senior level',
            'required_education': "Bachelor's Degree",
            'contact_email': 'careers@google.com'
        }
    },
    {
        'name': '3. Borderline/Vague Job (should trigger LLM)',
        'data': {
            'title': 'Customer Service Representative',
            'description': 'We need people for our customer service team. Work from home. Good pay.',
            'requirements': 'Must be available immediately.',
            'company_profile': 'A growing company.',
            'benefits': 'Good salary, flexible schedule',
            'salary_range': None,
            'location': None,
            'employment_type': 'Full-time',
            'contact_email': 'hiring@newcompany123.com'
        }
    },
    {
        'name': '4. Subtle Scam (MLM vibes)',
        'data': {
            'title': 'Independent Marketing Consultant',
            'description': 'Join our team of successful marketers! No experience needed. We train you. Unlimited income potential. Build your own team and earn commissions from their sales too!',
            'requirements': 'Self-motivated individuals. Must invest $200 in starter package.',
            'company_profile': 'Global Wellness Corp - Empowering entrepreneurs since 2020. Join thousands of success stories!',
            'benefits': 'Work from anywhere, unlimited income, bonus trips, car allowance',
            'salary_range': '$5,000-$50,000/month',
            'location': None,
            'employment_type': 'Contract',
            'contact_email': 'success@globalwellness.biz'
        }
    },
]

print("\n" + "=" * 70)
print("  RUNNING TEST CASES")
print("=" * 70)

# Run twice: once with default thresholds, once with "always use LLM"
runs = [
    ("RUN A: Default thresholds (LLM for 0.25-0.75 only)", 0.25, 0.75),
    ("RUN B: Always use LLM (force Gemini on every job)", 0.0, 1.0),
]

for run_name, low, high in runs:
    print(f"\n{'#'*70}")
    print(f"  {run_name}")
    print(f"{'#'*70}")
    
    for test in test_cases:
        print(f"\n{'='*70}")
        print(f"  {test['name']}")
        print(f"  Title: {test['data']['title']}")
        print(f"{'='*70}")
        
        result = hybrid_predict(test['data'], uncertainty_low=low, uncertainty_high=high)
        
        print(f"\n  [MODEL]  Probability: {result['model_probability']:.2%}")
        print(f"  [FINAL]  Probability: {result['final_probability']:.2%}")
        print(f"  [RISK]   Level: {result['risk_level']}")
        print(f"  [FAKE?]  {result['is_fake']}")
        print(f"  [METHOD] {result['method']}")
        print(f"  [TIME]   Model: {result['model_time_ms']}ms, Total: {result['total_time_ms']}ms")
        
        gf = result['graph_features']
        print(f"  [GRAPH]  Trust: {gf['domain_trust']:.2f}, Free email: {gf['email_free']}, "
              f"LinkedIn: {gf['has_linkedin']}, Suspicious: {gf['suspicious_domain']}")
        
        if result['used_llm']:
            llm = result['llm_analysis']
            print(f"\n  [LLM]    Provider: {llm['provider']}")
            print(f"  [LLM]    Probability: {llm['probability']:.2%}")
            print(f"  [LLM]    Confidence: {llm['confidence']}")
            print(f"  [LLM]    Cost: ${llm['cost']:.4f} | Cached: {llm['cached']} | Time: {llm['time_ms']}ms")
            if llm['red_flags']:
                print(f"  [LLM]    Red flags:")
                for flag in llm['red_flags']:
                    print(f"             - {flag}")

print("\n" + "=" * 70)
print("  PIPELINE TEST COMPLETE")
print("=" * 70)

# Summary
print("\nSummary:")
print(f"  Model: Phase 2 (DistilBERT + Metadata + Graph)")
print(f"  LLM:   {'Gemini (free)' if llm_available else 'Not available'}")
print(f"  Tests: {len(test_cases)} jobs analyzed")
print(f"  LLM triggers when model probability is between 0.25 and 0.75")
