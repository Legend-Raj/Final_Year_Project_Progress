"""
Quick test for trained model
Run this after copying best_model_full.pt to models/ folder
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.predict import FakeJobPredictor

# Initialize predictor
print("Loading model...")
predictor = FakeJobPredictor()

# Test 1: Normal job (should be LOW risk)
print("\n" + "="*60)
print("TEST 1: Normal Marketing Job")
print("="*60)
normal_job = {
    'title': 'Marketing Intern',
    'description': 'We are looking for a marketing intern to help with social media campaigns and content creation. You will work with our marketing team to develop engaging content.',
    'requirements': 'Currently enrolled in college, marketing or communications major preferred. Strong writing skills.',
    'company_profile': 'We are a fast-growing tech startup based in New York. We have been in business for 5 years and have 50+ employees.',
    'benefits': 'Flexible hours, learning opportunities, potential for full-time conversion',
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
    'function': 'Marketing'
}

result = predictor.predict(normal_job)
print(f"Fraud Probability: {result['fraud_probability']:.4f} ({result['fraud_probability']*100:.1f}%)")
print(f"Risk Level: {result['risk_level']}")
print(f"Is Fake: {result['is_fake']}")

# Test 2: Suspicious job (should be HIGH risk)
print("\n" + "="*60)
print("TEST 2: Suspicious Job (Too good to be true)")
print("="*60)
suspicious_job = {
    'title': 'Work From Home - $500/Day Guaranteed!',
    'description': 'No experience needed! Earn $500 per day working from home. Send $50 for starter kit and training materials. Contact us at quickjobs@gmail.com',
    'requirements': 'No experience required. Must have bank account.',
    'company_profile': 'We are a new company offering amazing opportunities.',
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
    'function': None
}

result = predictor.predict(suspicious_job)
print(f"Fraud Probability: {result['fraud_probability']:.4f} ({result['fraud_probability']*100:.1f}%)")
print(f"Risk Level: {result['risk_level']}")
print(f"Is Fake: {result['is_fake']}")

# Test 3: Real job with salary
print("\n" + "="*60)
print("TEST 3: Software Engineer (Realistic)")
print("="*60)
real_job = {
    'title': 'Senior Software Engineer',
    'description': 'We are seeking an experienced software engineer to join our backend team. You will design and implement scalable APIs, work with microservices architecture, and mentor junior developers.',
    'requirements': '5+ years experience in Python, experience with cloud platforms (AWS/GCP), knowledge of databases and caching strategies.',
    'company_profile': 'TechCorp is a leading software company founded in 2010. We have offices in San Francisco, New York, and London. Our products serve millions of users worldwide.',
    'benefits': 'Competitive salary ($120k-160k), health insurance, 401k matching, flexible PTO, remote work options.',
    'salary_range': '$120,000 - $160,000',
    'department': 'Engineering',
    'has_company_logo': 1,
    'has_questions': 1,
    'telecommuting': 1,
    'location': 'US, CA, San Francisco',
    'employment_type': 'Full-time',
    'required_experience': 'Mid-Senior level',
    'required_education': "Bachelor's Degree",
    'industry': 'Computer Software',
    'function': 'Engineering'
}

result = predictor.predict(real_job)
print(f"Fraud Probability: {result['fraud_probability']:.4f} ({result['fraud_probability']*100:.1f}%)")
print(f"Risk Level: {result['risk_level']}")
print(f"Is Fake: {result['is_fake']}")

print("\n" + "="*60)
print("All tests completed!")
print("="*60)
