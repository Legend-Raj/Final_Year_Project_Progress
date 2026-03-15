"""
Inference script - predict on new job postings
"""
import torch
import pickle
import pandas as pd
from transformers import DistilBertTokenizer
from model import FakeJobDetector


class FakeJobPredictor:
    def __init__(self, model_path='models/best_model_full.pt', 
                 scaler_path='processed/scaler.pkl',
                 feature_names_path='processed/feature_names.pkl'):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load scaler and feature names
        self.scaler = pickle.load(open(scaler_path, 'rb'))
        self.feature_names = pickle.load(open(feature_names_path, 'rb'))
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Create model with correct dimensions
        num_meta_features = len(self.feature_names)
        
        self.model = FakeJobDetector(num_meta_features=num_meta_features)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        print(f"Model loaded. Best Val AUC: {checkpoint.get('best_val_auc', 'N/A')}")
    
    def extract_features_from_job(self, job_dict):
        """
        Extract metadata features from a job posting dictionary
        """
        features = {}
        
        # Same feature extraction as training
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
        
        # Employment type
        emp_type = job_dict.get('employment_type', 'nan')
        for fname in self.feature_names:
            if fname.startswith('emp_type_'):
                features[fname] = 1 if fname == f'emp_type_{emp_type}' else 0
        
        # Experience and education
        exp_mapping = {
            'Not Applicable': 0, 'Entry level': 1, 'Associate': 2,
            'Mid-Senior level': 3, 'Director': 4, 'Executive': 5
        }
        features['experience_level'] = exp_mapping.get(job_dict.get('required_experience'), 0)
        
        edu_mapping = {
            'Unspecified': 0, 'High School or equivalent': 1,
            'Vocational': 2, 'Some College Coursework Completed': 3,
            'Associate Degree': 4, "Bachelor's Degree": 5,
            "Master's Degree": 6, 'Doctorate': 7, 'Professional': 8
        }
        features['education_level'] = edu_mapping.get(job_dict.get('required_education'), 0)
        
        # Industry and Function
        industry = job_dict.get('industry', '')
        for fname in self.feature_names:
            if fname.startswith('industry_'):
                ind_name = fname.replace('industry_', '')
                features[fname] = 1 if industry == ind_name else 0
        
        function = job_dict.get('function', '')
        for fname in self.feature_names:
            if fname.startswith('function_'):
                func_name = fname.replace('function_', '')
                features[fname] = 1 if function == func_name else 0
        
        features['desc_missing'] = 1 if not job_dict.get('description') else 0
        features['requirements_missing'] = 1 if not job_dict.get('requirements') else 0
        
        # Ensure all features are present
        feature_vector = [features.get(f, 0) for f in self.feature_names]
        
        return feature_vector
    
    def predict(self, job_dict):
        """
        Predict if a job posting is fake
        """
        # Prepare text
        text = f"Title: {job_dict.get('title', '')}. "
        text += f"Description: {job_dict.get('description', '')}. "
        text += f"Requirements: {job_dict.get('requirements', '')}. "
        text += f"Company: {job_dict.get('company_profile', '')}. "
        text += f"Benefits: {job_dict.get('benefits', '')}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prepare metadata
        meta_features = self.extract_features_from_job(job_dict)
        meta_scaled = self.scaler.transform([meta_features])
        meta_tensor = torch.tensor(meta_scaled, dtype=torch.float32).to(self.device)
        
        # Predict
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            prob = self.model.predict_proba(input_ids, attention_mask, meta_tensor)
        
        prob_value = prob.item()
        
        # Risk level (adjusted based on evaluation - best threshold is ~0.10)
        if prob_value < 0.05:
            risk = "LOW"
        elif prob_value < 0.15:
            risk = "MEDIUM"
        else:
            risk = "HIGH"
        
        return {
            'fraud_probability': prob_value,
            'is_fake': prob_value > 0.10,  # Best threshold from evaluation
            'risk_level': risk,
            'confidence': abs(prob_value - 0.5) * 2  # 0 to 1
        }


# Example usage
if __name__ == "__main__":
    predictor = FakeJobPredictor()
    
    # Test with a sample job
    sample_job = {
        'title': 'Marketing Intern',
        'description': 'We are looking for a marketing intern to help with social media...',
        'requirements': 'Currently enrolled in college, marketing major preferred',
        'company_profile': 'We are a fast-growing tech startup...',
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
        'function': 'Marketing'
    }
    
    result = predictor.predict(sample_job)
    print(f"\nPrediction Result:")
    print(f"Fraud Probability: {result['fraud_probability']:.4f}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Is Fake: {result['is_fake']}")
