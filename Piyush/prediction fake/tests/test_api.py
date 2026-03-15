"""Quick test for the API"""
import requests
import json

BASE = "http://localhost:8000"

# Health check
print("=== HEALTH CHECK ===")
r = requests.get(f"{BASE}/health")
print(json.dumps(r.json(), indent=2))

# Test 1: Scam job
print("\n=== TEST 1: Scam Job ===")
r = requests.post(f"{BASE}/predict", json={
    "title": "Work From Home - $500/Day Guaranteed!",
    "description": "No experience needed! Send $50 for starter kit.",
    "company_profile": "Quick money opportunities!",
    "contact_email": "quickmoney@gmail.com"
})
data = r.json()
prob = data["fraud_probability"]
print(f"  Fraud: {prob:.1%}")
print(f"  Risk: {data['risk_level']}")
print(f"  Method: {data['method']}")
print(f"  Time: {data['processing_time_ms']}ms")
llm = data.get("llm_result")
if llm:
    print(f"  LLM: {llm['confidence']} confidence, {llm['probability']:.1%}")
    for flag in llm["red_flags"][:3]:
        print(f"    - {flag}")
print(f"  Explanation: {data['explanation'][:200]}")

# Test 2: Legit job
print("\n=== TEST 2: Legit Job ===")
r = requests.post(f"{BASE}/predict", json={
    "title": "Senior Software Engineer",
    "description": "Join our engineering team. Build scalable distributed systems.",
    "requirements": "5+ years Python, CS degree required",
    "company_profile": "Google LLC, multinational tech company founded 1998.",
    "salary_range": "$150k-$200k",
    "location": "Mountain View, CA",
    "contact_email": "careers@google.com"
})
data = r.json()
prob = data["fraud_probability"]
print(f"  Fraud: {prob:.1%}")
print(f"  Risk: {data['risk_level']}")
print(f"  Method: {data['method']}")
print(f"  Time: {data['processing_time_ms']}ms")

# Test 3: MLM scam
print("\n=== TEST 3: MLM Scam ===")
r = requests.post(f"{BASE}/predict", json={
    "title": "Independent Marketing Consultant",
    "description": "Build your own team! Earn commissions! No experience needed. Invest $200 in starter package.",
    "company_profile": "Global Wellness Corp - Empowering entrepreneurs!",
    "salary_range": "$5,000-$50,000/month",
    "contact_email": "success@globalwellness.biz"
})
data = r.json()
prob = data["fraud_probability"]
print(f"  Fraud: {prob:.1%}")
print(f"  Risk: {data['risk_level']}")
print(f"  Method: {data['method']}")
print(f"  Time: {data['processing_time_ms']}ms")
llm = data.get("llm_result")
if llm:
    print(f"  LLM: {llm['confidence']} confidence, {llm['probability']:.1%}")
    for flag in llm["red_flags"][:3]:
        print(f"    - {flag}")

print("\n=== ALL TESTS DONE ===")
