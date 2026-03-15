"""
Final Fusion Model - Ultimate Architecture
Combines: Text + Metadata + Graph + LLM → Final Prediction
"""
import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional


class FinalFusionLayer(nn.Module):
    """
    Learned fusion layer combining all modalities
    
    Input: 11 features
    - model_probability (1)
    - model_confidence (1) 
    - llm_probability (1)
    - llm_confidence (1)
    - llm_num_red_flags (1)
    - llm_reasoning_score (1)
    - graph_domain_trust (1)
    - graph_email_type (1)
    - graph_linkedin_presence (1)
    - graph_suspicious_indicators (1)
    - metadata_richness_score (1)
    
    Output: Final fraud probability
    """
    
    def __init__(self, input_dim: int = 11, hidden_dim: int = 64):
        super().__init__()
        
        self.fusion = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 2
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Layer 3
            nn.Linear(32, 16),
            nn.ReLU(),
            
            # Final output
            nn.Linear(16, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 11] - all features concatenated
        Returns:
            logits: [batch] - raw output (apply sigmoid for probability)
        """
        return self.fusion(x).squeeze(-1)


class CompleteUltimatePredictor:
    """
    Ultimate predictor with learned final fusion layer
    
    Architecture:
    Phase 2 Model ──┐
    LLM Analysis ───┼──► Feature Extraction ──► FinalFusionLayer ──► Output
    Graph Features ─┤         (11-dim)
    Metadata ───────┘
    """
    
    def __init__(self, 
                 phase2_model_path: str = "models/phase2_best.pt",
                 fusion_model_path: Optional[str] = None,
                 use_llm: bool = True):
        
        import sys
        sys.path.append('src')
        
        from phase2.model_v2 import FakeJobDetectorV2
        from phase2.network_features import NetworkFeatureExtractor
        from phase3.llm_analyzer_v2 import LLMJobAnalyzer
        from transformers import DistilBertTokenizer
        import pickle
        
        print("=" * 70)
        print("ULTIMATE PREDICTOR - Final Fusion Architecture")
        print("=" * 70)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        # 1. Load Phase 2 Model (Text + Metadata + Graph encoder)
        print("\n[1/4] Loading Phase 2 model...")
        checkpoint = torch.load(phase2_model_path, map_location=self.device, weights_only=False)
        
        self.phase2_model = FakeJobDetectorV2(num_meta_features=41, num_graph_features=14)
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('model'))
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('bert.', 'distilbert.').replace('meta_enc.', 'metadata_encoder.').replace('graph_enc.', 'graph_encoder.encoder.')
            new_state_dict[new_key] = v
        self.phase2_model.load_state_dict(new_state_dict, strict=False)
        self.phase2_model.to(self.device)
        self.phase2_model.eval()
        print("Phase 2 model loaded!")
        
        # 2. Initialize Final Fusion Layer (LEARNED)
        print("\n[2/4] Initializing Final Fusion Layer...")
        self.fusion_layer = FinalFusionLayer(input_dim=11, hidden_dim=64).to(self.device)
        
        # Load fusion weights if available
        if fusion_model_path and os.path.exists(fusion_model_path):
            fusion_ckpt = torch.load(fusion_model_path, map_location=self.device, weights_only=False)
            self.fusion_layer.load_state_dict(fusion_ckpt['fusion_state_dict'])
            print("Fusion layer weights loaded!")
        else:
            print("Fusion layer initialized with random weights (train or use default)")
            # Set to evaluation mode with default reasonable weights
            self.fusion_layer.eval()
        
        # 3. LLM Analyzer
        print("\n[3/4] Setting up LLM analyzer...")
        self.use_llm = use_llm
        if use_llm:
            try:
                self.llm_analyzer = LLMJobAnalyzer()
                print("LLM analyzer ready!")
            except (ValueError, ImportError, RuntimeError) as e:
                print(f"LLM not available: {e}")
                self.use_llm = False
        
        # 4. Helper components
        print("\n[4/4] Loading helper components...")
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.net_extractor = NetworkFeatureExtractor()
        with open('processed/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        with open('processed/feature_names.pkl', 'rb') as f:
            self.feature_names = pickle.load(f)
        
        print("\n" + "=" * 70)
        print("All components loaded!")
        print("Architecture: Phase2 → [Features] → Fusion Layer → Final Output")
        print("=" * 70)
    
    def _extract_all_features(self, job_dict: dict) -> Dict:
        """Extract all 11 features for fusion layer"""
        
        # 1. Get Phase 2 model prediction
        text = f"Title: {job_dict.get('title', '')}. Description: {job_dict.get('description', '')}. "
        text += f"Requirements: {job_dict.get('requirements', '')}. Company: {job_dict.get('company_profile', '')}"
        
        encoding = self.tokenizer(text, max_length=512, padding='max_length', 
                                  truncation=True, return_tensors='pt')
        
        # Metadata
        meta_features = self._get_metadata_features(job_dict)
        meta_tensor = torch.tensor([meta_features], dtype=torch.float32).to(self.device)
        
        # Graph
        graph_obj = self.net_extractor.extract(
            company_profile=job_dict.get('company_profile', ''),
            contact_email=job_dict.get('contact_email')
        )
        graph_vector = graph_obj.to_vector()
        graph_tensor = torch.tensor([graph_vector], dtype=torch.float32).to(self.device)
        
        # Model prediction
        with torch.no_grad():
            model_prob = self.phase2_model.predict_proba(
                encoding['input_ids'].to(self.device),
                encoding['attention_mask'].to(self.device),
                meta_tensor,
                graph_tensor
            ).item()
        
        # Model confidence (distance from 0.5)
        model_confidence = abs(model_prob - 0.5) * 2  # 0 to 1
        
        # 2. LLM features (if available)
        llm_prob = model_prob  # default fallback
        llm_confidence = 0.0
        llm_num_red_flags = 0
        llm_reasoning_score = 0.5
        
        if self.use_llm and 0.25 <= model_prob <= 0.75:
            llm_result = self.llm_analyzer.analyze(job_dict, model_prob)
            llm_prob = llm_result.fraud_probability
            llm_confidence = 1.0 if llm_result.confidence == 'high' else 0.5 if llm_result.confidence == 'medium' else 0.25
            llm_num_red_flags = len(llm_result.red_flags)
            # Simple sentiment score from reasoning length
            llm_reasoning_score = min(1.0, len(llm_result.reasoning) / 500)
        
        # 3. Graph features (individual)
        graph_domain_trust = graph_obj.domain_trust_score
        graph_email_type = 1.0 if graph_obj.email_is_corporate else 0.5 if not graph_obj.email_is_free_provider else 0.0
        graph_linkedin = float(graph_obj.has_linkedin_page)
        graph_suspicious = float(graph_obj.is_known_fake_domain)
        
        # 4. Metadata richness (how complete is the job posting)
        filled_fields = sum([
            bool(job_dict.get('description')),
            bool(job_dict.get('requirements')),
            bool(job_dict.get('company_profile')),
            bool(job_dict.get('salary_range')),
            bool(job_dict.get('location'))
        ])
        metadata_richness = filled_fields / 5.0
        
        # Combine all 11 features
        features = {
            'model_probability': model_prob,
            'model_confidence': model_confidence,
            'llm_probability': llm_prob,
            'llm_confidence': llm_confidence,
            'llm_num_red_flags': min(llm_num_red_flags, 5) / 5.0,  # Normalize to 0-1
            'llm_reasoning_score': llm_reasoning_score,
            'graph_domain_trust': graph_domain_trust,
            'graph_email_type': graph_email_type,
            'graph_linkedin_presence': graph_linkedin,
            'graph_suspicious': graph_suspicious,
            'metadata_richness': metadata_richness
        }
        
        return features
    
    def _get_metadata_features(self, job_dict: dict) -> np.ndarray:
        """Extract metadata features using shared utility"""
        from phase3.feature_utils import extract_metadata_from_job
        feature_vector = extract_metadata_from_job(job_dict, self.feature_names)
        return self.scaler.transform([feature_vector])[0]
    
    def predict(self, job_dict: dict, use_learned_fusion: bool = True) -> Dict:
        """
        Ultimate prediction with final fusion layer
        
        Args:
            job_dict: Job posting data
            use_learned_fusion: If True, use neural fusion layer, else weighted average
        
        Returns:
            Complete prediction result with all features
        """
        # Extract all features
        features = self._extract_all_features(job_dict)
        
        # Create feature vector for fusion layer
        feature_list = [
            features['model_probability'],
            features['model_confidence'],
            features['llm_probability'],
            features['llm_confidence'],
            features['llm_num_red_flags'],
            features['llm_reasoning_score'],
            features['graph_domain_trust'],
            features['graph_email_type'],
            features['graph_linkedin_presence'],
            features['graph_suspicious'],
            features['metadata_richness']
        ]
        
        if use_learned_fusion:
            # Use learned neural fusion layer
            fusion_input = torch.tensor([feature_list], dtype=torch.float32).to(self.device)
            with torch.no_grad():
                final_logit = self.fusion_layer(fusion_input)
                final_prob = torch.sigmoid(final_logit).item()
            method = "learned_fusion"
        else:
            # Fallback to weighted average
            weights = [0.4, 0.0, 0.4, 0.0, 0.05, 0.0, 0.05, 0.05, 0.03, 0.02, 0.0]
            final_prob = sum(f * w for f, w in zip(feature_list, weights))
            method = "weighted_average"
        
        # Risk level
        if final_prob < 0.3:
            risk = "LOW"
        elif final_prob < 0.7:
            risk = "MEDIUM"
        else:
            risk = "HIGH"
        
        return {
            'final_probability': final_prob,
            'final_risk': risk,
            'is_fake': final_prob > 0.5,
            'method': method,
            'features': features,
            'feature_vector': feature_list,
            'explanation': self._generate_explanation(features, final_prob)
        }
    
    def _generate_explanation(self, features: Dict, final_prob: float) -> str:
        """Generate human-readable explanation"""
        parts = []
        
        # Model contribution
        model_diff = abs(features['model_probability'] - final_prob)
        parts.append(f"Model prediction: {features['model_probability']:.1%}")
        
        # LLM contribution
        if features['llm_confidence'] > 0:
            parts.append(f"LLM analysis: {features['llm_probability']:.1%} (confidence: {features['llm_confidence']:.0%})")
            if features['llm_num_red_flags'] > 0:
                parts.append(f"Red flags detected: {int(features['llm_num_red_flags'] * 5)}")
        
        # Graph indicators
        if features['graph_suspicious'] > 0:
            parts.append("Suspicious domain patterns detected")
        if features['graph_email_type'] < 0.5:
            parts.append("Free email provider used")
        if features['graph_linkedin_presence'] < 0.5:
            parts.append("No LinkedIn presence")
        
        # Final
        parts.append(f"\nFinal assessment: {final_prob:.1%} - {('FAKE' if final_prob > 0.5 else 'LEGITIMATE')}")
        
        return "\n".join(parts)


if __name__ == "__main__":
    # Example usage
    predictor = CompleteUltimatePredictor(use_llm=False)  # Set True if API key available
    
    test_job = {
        'title': 'Work From Home - $500/Day Guaranteed!',
        'description': 'No experience needed! Send $50 for starter kit. Earn easy money!',
        'requirements': 'No requirements',
        'company_profile': 'Quick money opportunities! Contact now!',
        'benefits': 'Unlimited earnings!',
        'contact_email': 'quickmoney@gmail.com'
    }
    
    result = predictor.predict(test_job)
    print(f"\nFinal Probability: {result['final_probability']:.2%}")
    print(f"Risk: {result['final_risk']}")
    print(f"Method: {result['method']}")
    print(f"\n{result['explanation']}")
