"""
Complete Hybrid Predictor
Combines Phase 2 Model + Calibration + LLM for production use
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pickle
import numpy as np
from transformers import DistilBertTokenizer

from phase2.model_v2 import FakeJobDetectorV2
from phase2.network_features import NetworkFeatureExtractor
from phase3.llm_analyzer_v2 import LLMJobAnalyzer  # V2: supports Gemini (free)
from phase3.feature_utils import extract_metadata_from_job


# Optimal raw threshold from evaluation (maximizes F1 without calibration)
RAW_OPTIMAL_THRESHOLD = 0.03


class CompleteHybridPredictor:
    """
    Production-ready hybrid predictor with calibration
    
    Workflow:
    1. Phase 2 Model predicts raw probability (compressed to ~0-0.15 range)
    2. Calibrator transforms raw prob to well-calibrated probability (0-1 range)
    3. If uncertain (raw >= 0.03 AND calibrated < 0.85), call LLM
    4. Combine calibrated + LLM predictions
    5. Return detailed result with 0.5 as the natural threshold
    """
    
    def __init__(self, 
                 model_path: str = "models/phase2_best.pt",
                 scaler_path: str = "processed/scaler.pkl",
                 feature_names_path: str = "processed/feature_names.pkl",
                 calibrator_path: str = "models/calibrator.pkl",
                 use_llm: bool = True,
                 uncertainty_low: float = 0.03,
                 uncertainty_high: float = 0.85):
        
        print("=" * 70)
        print("COMPLETE HYBRID PREDICTOR")
        print("Phase 2 Model + Calibration + LLM Integration")
        print("=" * 70)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        # Load scaler and feature names
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        with open(feature_names_path, 'rb') as f:
            self.feature_names = pickle.load(f)
        
        # Load Phase 2 model
        print("\nLoading Phase 2 model...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.model = FakeJobDetectorV2(
            num_meta_features=len(self.feature_names),
            num_graph_features=14
        )
        
        # Load weights with key mapping
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('model'))
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('bert.', 'distilbert.')
            new_key = new_key.replace('meta_enc.', 'metadata_encoder.')
            new_key = new_key.replace('graph_enc.', 'graph_encoder.encoder.')
            new_state_dict[new_key] = v
        
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
        
        # Load calibrator (transforms raw 0-0.15 range to calibrated 0-1 range)
        self.calibrator = None
        if os.path.exists(calibrator_path):
            with open(calibrator_path, 'rb') as f:
                self.calibrator = pickle.load(f)
            print(f"Calibrator loaded (primary: {self.calibrator['primary']})")
        else:
            print("Warning: No calibrator found. Run: python src/calibrate.py")
            print("Raw model outputs will be used directly (suboptimal).")
        
        # Initialize tokenizer and network extractor
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.net_extractor = NetworkFeatureExtractor()
        
        # LLM setup
        self.use_llm = use_llm
        self.uncertainty_low = uncertainty_low    # Raw model threshold
        self.uncertainty_high = uncertainty_high  # Calibrated prob ceiling
        
        if use_llm:
            try:
                self.llm_analyzer = LLMJobAnalyzer()
                print("LLM analyzer initialized")
            except ValueError as e:
                print(f"Warning: {e}")
                print("Running without LLM")
                self.use_llm = False
        
        print("=" * 70)
    
    def calibrate_prob(self, raw_prob: float) -> float:
        """Apply calibration to transform raw model output to well-calibrated probability."""
        if self.calibrator is None:
            return raw_prob
        
        method = self.calibrator['primary']  # 'isotonic' or 'platt'
        if method == 'isotonic':
            return float(self.calibrator['isotonic'].predict([raw_prob])[0])
        else:
            return float(self.calibrator['platt'].predict_proba([[raw_prob]])[0][1])
    
    def _extract_metadata(self, job_dict: dict) -> np.ndarray:
        """Extract metadata features using shared utility"""
        feature_vector = extract_metadata_from_job(job_dict, self.feature_names)
        return self.scaler.transform([feature_vector])[0]
    
    def predict(self, job_dict: dict) -> dict:
        """
        Complete prediction pipeline with calibration
        
        Returns:
            {
                'raw_probability': raw model output (0-0.15 range),
                'calibrated_probability': calibrated probability (0-1 range),
                'final_probability': final fraud probability after LLM fusion,
                'is_fake': boolean (threshold = 0.5 on calibrated),
                'risk_level': 'LOW'/'MEDIUM'/'HIGH',
                'method': 'model_only' or 'hybrid',
                'llm_analysis': {details if LLM used},
                'explanation': human readable explanation
            }
        """
        # Prepare text
        text = f"Title: {job_dict.get('title', '')}. "
        text += f"Description: {job_dict.get('description', '')}. "
        text += f"Requirements: {job_dict.get('requirements', '')}. "
        text += f"Company: {job_dict.get('company_profile', '')}. "
        text += f"Benefits: {job_dict.get('benefits', '')}"
        
        encoding = self.tokenizer(text, max_length=512, padding='max_length',
                                  truncation=True, return_tensors='pt')
        
        # Extract features
        meta = self._extract_metadata(job_dict)
        meta_tensor = torch.tensor([meta], dtype=torch.float32).to(self.device)
        
        graph_obj = self.net_extractor.extract(
            company_profile=job_dict.get('company_profile', ''),
            contact_email=job_dict.get('contact_email'),
            company_name=job_dict.get('company_name', job_dict.get('title', ''))
        )
        graph_tensor = torch.tensor([graph_obj.to_vector()], dtype=torch.float32).to(self.device)
        
        # Model prediction (raw)
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            raw_prob = self.model.predict_proba(input_ids, attention_mask, 
                                                 meta_tensor, graph_tensor).item()
        
        # Apply calibration (transforms 0-0.15 -> 0-1 range)
        cal_prob = self.calibrate_prob(raw_prob)
        
        # Initialize result
        result = {
            'raw_probability': raw_prob,
            'calibrated_probability': cal_prob,
            'model_probability': raw_prob,  # backwards compat
            'final_probability': cal_prob,
            'method': 'model_only',
            'used_llm': False,
            'llm_analysis': None,
            'graph_features': {
                'domain_trust_score': graph_obj.domain_trust_score,
                'email_is_free': bool(graph_obj.email_is_free_provider),
                'has_linkedin': bool(graph_obj.has_linkedin_page)
            }
        }
        
        # LLM triggering logic:
        # - Raw prob >= 0.03 means model thinks it's potentially suspicious
        # - Calibrated prob < 0.85 means model isn't highly confident it's fraud
        # - LLM helps in this uncertain zone
        llm_trigger = raw_prob >= RAW_OPTIMAL_THRESHOLD and cal_prob < self.uncertainty_high
        
        if self.use_llm and llm_trigger:
            print(f"Model uncertain (raw={raw_prob:.2%}, cal={cal_prob:.2%}), consulting LLM...")
            
            try:
                llm_result = self.llm_analyzer.analyze(job_dict, cal_prob)
                
                # Weighted combination based on LLM confidence
                if llm_result.confidence == "high":
                    final_prob = 0.3 * cal_prob + 0.7 * llm_result.fraud_probability
                elif llm_result.confidence == "medium":
                    final_prob = 0.5 * cal_prob + 0.5 * llm_result.fraud_probability
                else:
                    final_prob = 0.7 * cal_prob + 0.3 * llm_result.fraud_probability
                
                result['final_probability'] = final_prob
                result['method'] = 'hybrid'
                result['used_llm'] = True
                result['llm_analysis'] = {
                    'probability': llm_result.fraud_probability,
                    'confidence': llm_result.confidence,
                    'reasoning': llm_result.reasoning,
                    'red_flags': llm_result.red_flags,
                    'recommendations': llm_result.recommendations,
                    'cost_usd': llm_result.cost_usd,
                    'cached': llm_result.cached
                }
                
                result['explanation'] = (
                    f"Model (raw={raw_prob:.1%}, calibrated={cal_prob:.1%}), "
                    f"LLM ({llm_result.confidence}): {llm_result.fraud_probability:.1%}. "
                    f"Final: {final_prob:.1%}. "
                    f"Issues: {', '.join(llm_result.red_flags[:3]) if llm_result.red_flags else 'None'}"
                )
            except Exception as e:
                print(f"LLM failed: {e}")
                result['explanation'] = (
                    f"Model: raw={raw_prob:.1%}, calibrated={cal_prob:.1%}. LLM failed: {e}"
                )
        else:
            if cal_prob >= self.uncertainty_high:
                result['explanation'] = f"Model highly confident fraud: {cal_prob:.1%} (raw: {raw_prob:.1%})"
            elif raw_prob < RAW_OPTIMAL_THRESHOLD:
                result['explanation'] = f"Model confident legitimate: {cal_prob:.1%} (raw: {raw_prob:.1%})"
            else:
                result['explanation'] = f"Model prediction: {cal_prob:.1%} (raw: {raw_prob:.1%})"
        
        # Risk level based on final (calibrated + LLM) probability
        prob = result['final_probability']
        if prob < 0.3:
            result['risk_level'] = 'LOW'
        elif prob < 0.7:
            result['risk_level'] = 'MEDIUM'
        else:
            result['risk_level'] = 'HIGH'
        
        result['is_fake'] = prob > 0.5
        
        return result


if __name__ == "__main__":
    # Example usage
    predictor = CompleteHybridPredictor()
    
    test_job = {
        'title': 'Marketing Intern',
        'description': 'Join our marketing team for an exciting internship opportunity.',
        'requirements': 'Currently enrolled in college',
        'company_profile': 'TechStartup Inc - A growing company founded in 2018',
        'benefits': 'Flexible hours',
        'contact_email': 'jobs@techstartup.com'
    }
    
    result = predictor.predict(test_job)
    print(f"\nRaw probability:        {result['raw_probability']:.2%}")
    print(f"Calibrated probability: {result['calibrated_probability']:.2%}")
    print(f"Final probability:      {result['final_probability']:.2%}")
    print(f"Risk level:             {result['risk_level']}")
    print(f"Is fake:                {result['is_fake']}")
    print(f"Method:                 {result['method']}")
    print(f"Explanation:            {result['explanation']}")
