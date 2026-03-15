"""
Phase 2 Complete Pipeline
Single script to run entire Phase 2 workflow
"""
import os
import sys
import argparse

sys.path.append('src')


def check_phase1_data():
    """Check if Phase 1 data exists"""
    required = [
        'processed/train.pkl',
        'processed/val.pkl',
        'processed/test.pkl',
        'processed/scaler.pkl',
        'processed/feature_names.pkl',
        'models/best_model_full.pt'
    ]
    
    missing = []
    for f in required:
        if not os.path.exists(f):
            missing.append(f)
    
    if missing:
        print("❌ Missing Phase 1 files:")
        for f in missing:
            print(f"   - {f}")
        print("\nRun Phase 1 first or download model from Colab")
        return False
    
    print("✅ Phase 1 data found")
    return True


def step1_enrich():
    """Step 1: Enrich data with graph features"""
    print("\n" + "="*60)
    print("STEP 1: Enriching Phase 1 data with graph features")
    print("="*60)
    
    from phase2.data_enrichment import DataEnricher
    
    enricher = DataEnricher()
    enricher.enrich_all_splits('processed')
    
    print("\n✅ Step 1 complete!")


def step2_train(epochs=10, use_gpu=True):
    """Step 2: Train Phase 2 model"""
    print("\n" + "="*60)
    print("STEP 2: Training Phase 2 model")
    print("="*60)
    
    import torch
    from phase2.train_phase2 import main as train_main
    import sys
    
    # Build args
    args = [
        '--processed-dir', 'processed',
        '--use-phase1', 'models/best_model_full.pt',
        '--epochs', str(epochs),
        '--batch-size', '16'
    ]
    
    if not use_gpu or not torch.cuda.is_available():
        print("⚠️ Training on CPU (will be slow)")
    
    sys.argv = ['train_phase2.py'] + args
    train_main()
    
    print("\n✅ Step 2 complete!")


def step3_test():
    """Step 3: Test the model"""
    print("\n" + "="*60)
    print("STEP 3: Testing Phase 2 model")
    print("="*60)
    
    from phase2.predict_phase2 import FakeJobPredictorV2
    
    predictor = FakeJobPredictorV2()
    
    # Test cases
    test_jobs = [
        {
            'name': 'Normal Marketing Job',
            'data': {
                'title': 'Marketing Intern',
                'description': 'Join our marketing team...',
                'requirements': 'Marketing major preferred',
                'company_profile': 'TechStartup Inc. is a growing company founded in 2018.',
                'benefits': 'Flexible hours',
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
        },
        {
            'name': 'Suspicious Job',
            'data': {
                'title': 'Work From Home - $500/Day!',
                'description': 'No experience needed! Send $50 for starter kit.',
                'requirements': 'No requirements',
                'company_profile': 'New opportunity for everyone!',
                'benefits': 'Unlimited earnings!',
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
        }
    ]
    
    for test in test_jobs:
        print(f"\n--- {test['name']} ---")
        result = predictor.predict(test['data'])
        print(f"Probability: {result['fraud_probability']:.2%}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Is Fake: {result['is_fake']}")
    
    print("\n✅ Step 3 complete!")


def main():
    parser = argparse.ArgumentParser(description='Phase 2 Pipeline')
    parser.add_argument('--step', choices=['enrich', 'train', 'test', 'all'], 
                       default='all', help='Which step to run')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--cpu', action='store_true', help='Force CPU training')
    
    args = parser.parse_args()
    
    print("="*60)
    print("FAKE JOB DETECTION - PHASE 2 PIPELINE")
    print("="*60)
    
    # Check Phase 1 data
    if not check_phase1_data():
        return 1
    
    # Run steps
    if args.step in ['enrich', 'all']:
        step1_enrich()
    
    if args.step in ['train', 'all']:
        step2_train(epochs=args.epochs, use_gpu=not args.cpu)
    
    if args.step in ['test', 'all']:
        if not os.path.exists('models/best_model_phase2.pt'):
            print("\n⚠️ Model not found. Train first or check path.")
        else:
            step3_test()
    
    print("\n" + "="*60)
    print("PHASE 2 COMPLETE!")
    print("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
