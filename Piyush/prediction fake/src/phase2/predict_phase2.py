"""
Phase 2 Inference Script
Includes graph/network feature extraction during prediction
"""
import os
import sys
import pickle
from typing import Dict, Optional

import torch
import numpy as np
from transformers import DistilBertTokenizer

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2.model_v2 import FakeJobDetectorV2
from phase2.network_features import NetworkFeatureExtractor
from phase2.config import get_config


class FakeJobPredictorV2:
    """
    Phase 2 Predictor with Graph Features
    
    Usage:
        predictor = FakeJobPredictorV2()
        result = predictor.predict(job_dict)
    """
    
    def __init__(self, 
                 model_path: str = 'models/best_model_phase2.pt',
                 scaler_path: str = 'processed/scaler.pkl',
                 feature_names_path: str = 'processed/feature_names.pkl',
                 device: Optional[str] = None):
        """
        Initialize Phase 2 predictor
        
        Args:
            model_path: Path to trained Phase 2 model
            scaler_path: Path to metadata scaler
            feature_names_path: Path to feature names
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.config = get_config()
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        print(f"Loading Phase 2 model on {self.device}...")
        
        # Load scaler and feature names
        self.scaler = pickle.load(open(scaler_path, 'rb'))
        self.feature_names = pickle.load(open(feature_names_path, 'rb'))
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Get dimensions
        num_meta_features = len(self.feature_names)
        
        # Create model
        self.model = FakeJobDetectorV2(
            num_meta_features=num_meta_features,
            num_graph_features=14,  # NetworkFeatures output
            graph_embedding_dim=32
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Initialize network feature extractor
        self.network_extractor = NetworkFeatureExtractor()
        
        # Get metrics from checkpoint
        metrics = checkpoint.get('metrics', {})
        self.best_threshold = metrics.get('threshold', 0.10)
        self.val_auc = checkpoint.get('best_val_auc', metrics.get('auc', 0.0))
        
        print(f"Model loaded. Best Val AUC: {self.val_auc:.4f}")
        print(f"Using threshold: {self.best_threshold:.2f}")
        
        # Show feature importance
        importance = self.model.get_feature_importance()
        print(f"\nFeature Importance:")
        print(f"  Text:    {importance['text']:.4f}")
        print(f"  Metadata: {importance['metadata']:.4f}")
        print(f"  Graph:   {importance['graph']:.4f}")
    
    def extract_metadata_features(self, job_dict: Dict) -> np.ndarray:
        """Extract metadata features (same as Phase 1)"""
        import pandas as pd
        
        features = {}
        
        # Text lengths
        features['desc_length'] = len(str(job_dict.get('description', '')))
        features['title_length'] = len(str(job_dict.get('title', '')))
        features['req_length'] = len(str(job_dict.get('requirements', '')))
        features['company_profile_length'] = len(str(job_dict.get('company_profile', '')))
        
        # Binary features
        features['has_salary'] = 1 if job_dict.get('salary_range') else 0
        features['has_department'] = 1 if job_dict.get('department') else 0
        features['has_company_logo'] = int(job_dict.get('has_company_logo', 0))
        features['has_questions'] = int(job_dict.get('has_questions', 0))
        features['telecommuting'] = int(job_dict.get('telecommuting', 0))
        
        # Location
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
        
        # Missing indicators
        features['desc_missing'] = 1 if not job_dict.get('description') else 0
        features['requirements_missing'] = 1 if not job_dict.get('requirements') else 0
        
        # Create feature vector
        feature_vector = [features.get(f, 0) for f in self.feature_names]
        
        # Scale
        return self.scaler.transform([feature_vector])[0]
    
    def predict(self, job_dict: Dict, return_features: bool = False) -> Dict:
        """
        Predict if a job posting is fake
        
        Args:
            job_dict: Job posting dictionary
            return_features: If True, return extracted features
            
        Returns:
            Dictionary with prediction results
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
        
        # Extract metadata features
        meta_features = self.extract_metadata_features(job_dict)
        meta_tensor = torch.tensor([meta_features], dtype=torch.float32).to(self.device)
        
        # Extract graph features (NEW in Phase 2)
        graph_features_obj = self.network_extractor.extract(
            company_profile=job_dict.get('company_profile', ''),
            contact_email=job_dict.get('contact_email'),
            company_name=job_dict.get('company_name', job_dict.get('title', ''))
        )
        graph_vector = graph_features_obj.to_vector()
        graph_tensor = torch.tensor([graph_vector], dtype=torch.float32).to(self.device)
        
        # Predict
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            prob = self.model.predict_proba(
                input_ids, 
                attention_mask, 
                meta_tensor,
                graph_tensor
            )
        
        prob_value = prob.item()
        
        # Risk level based on best threshold
        if prob_value < self.best_threshold * 0.5:
            risk = "LOW"
        elif prob_value < self.best_threshold * 1.5:
            risk = "MEDIUM"
        else:
            risk = "HIGH"
        
        result = {
            'fraud_probability': prob_value,
            'is_fake': prob_value > self.best_threshold,
            'risk_level': risk,
            'confidence': abs(prob_value - self.best_threshold) / max(self.best_threshold, 1 - self.best_threshold),
            'threshold_used': self.best_threshold
        }
        
        if return_features:
            result['features'] = {
                'metadata': meta_features.tolist(),
                'graph': graph_vector.tolist(),
                'graph_details': {
                    'domain_trust_score': graph_features_obj.domain_trust_score,
                    'domain_age_days': graph_features_obj.domain_age_days,
                    'email_is_corporate': graph_features_obj.email_is_corporate,
                    'email_is_free_provider': graph_features_obj.email_is_free_provider,
                    'has_linkedin_page': graph_features_obj.has_linkedin_page,
                    'is_known_fake_domain': graph_features_obj.is_known_fake_domain,
                }
            }
        
        return result
    
    def explain_prediction(self, job_dict: Dict) -> str:
        """
        Generate human-readable explanation for prediction
        """
        result = self.predict(job_dict, return_features=True)
        
        explanation = []
        explanation.append(f"Fraud Probability: {result['fraud_probability']:.2%}")
        explanation.append(f"Risk Level: {result['risk_level']}")
        explanation.append("")
        
        if 'features' in result:
            graph = result['features']['graph_details']
            explanation.append("Key Network Indicators:")
            
            if graph['is_known_fake_domain']:
                explanation.append("  ⚠️ Domain matches known fake patterns")
            
            if graph['email_is_free_provider'] and not graph['email_is_corporate']:
                explanation.append("  ⚠️ Using free email provider (Gmail/Yahoo)")
            
            if graph['domain_trust_score'] < 0.3:
                explanation.append(f"  ⚠️ Low domain trust score ({graph['domain_trust_score']:.2f})")
            elif graph['domain_trust_score'] > 0.7:
                explanation.append(f"  ✓ High domain trust score ({graph['domain_trust_score']:.2f})")
            
            if graph['has_linkedin_page']:
                explanation.append("  ✓ Has LinkedIn company page")
            else:
                explanation.append("  ⚠️ No LinkedIn presence detected")
        
        return "\n".join(explanation)


# Example usage
if __name__ == "__main__":
    predictor = FakeJobPredictorV2()
    
    # Test job
    test_job = {
        'title': 'Marketing Intern',
        'description': 'We are looking for a marketing intern...',
        'requirements': 'Currently enrolled in college...',
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
        'function': 'Marketing',
        'contact_email': 'jobs@techstartup.com'
    }
    
    result = predictor.predict(test_job, return_features=True)
    print("\nPrediction Result:")
    print(f"Probability: {result['fraud_probability']:.4f}")
    print(f"Risk: {result['risk_level']}")
    print(f"Is Fake: {result['is_fake']}")
    
    print("\n" + predictor.explain_prediction(test_job))
