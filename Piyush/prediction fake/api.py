"""
JobTrust REST API
FastAPI server with hybrid Phase 2 Model + Gemini LLM
"""
import sys
import os
import time
import logging

sys.path.append('src')

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import torch
import pickle
import numpy as np
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from transformers import DistilBertTokenizer
from phase2.model_v2 import FakeJobDetectorV2
from phase2.network_features import NetworkFeatureExtractor
from phase3.llm_analyzer_v2 import LLMJobAnalyzer
from phase3.final_fusion_model import FinalFusionLayer
from phase3.mock_llm import MockLLMAnalyzer
from phase3.feature_utils import extract_metadata_from_job

# Suppress noisy logs
logging.getLogger("phase2.model_v2").setLevel(logging.WARNING)
logging.getLogger("phase2.network_features").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ===================================================================
# Global model objects (loaded once at startup)
# ===================================================================
model = None
tokenizer = None
scaler = None
feature_names = None
net_extractor = None
llm_analyzer = None
calibrator = None       # Isotonic/Platt calibrator for raw model outputs
fusion_model = None     # Trained neural fusion layer (11-dim -> 1)
mock_llm = None         # Mock LLM for instant heuristic features (used by fusion)
device = None


def load_models():
    """Load all models and components into memory"""
    global model, tokenizer, scaler, feature_names, net_extractor, llm_analyzer, calibrator, fusion_model, mock_llm, device

    logger.info("Loading models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Load scaler and feature names
    with open('processed/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('processed/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)

    # Load Phase 2 model
    checkpoint = torch.load('models/phase2_best.pt', map_location=device, weights_only=False)
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
    logger.info("Phase 2 model loaded!")

    # Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Network feature extractor
    net_extractor = NetworkFeatureExtractor()

    # LLM (Gemini)
    try:
        llm_analyzer = LLMJobAnalyzer(provider="auto")
        logger.info(f"LLM ready: {llm_analyzer.provider}")
    except Exception as e:
        logger.warning(f"LLM not available: {e}")
        llm_analyzer = None

    # Calibrator (transforms raw model probs to well-calibrated probs)
    cal_path = 'models/calibrator.pkl'
    if os.path.exists(cal_path):
        with open(cal_path, 'rb') as f:
            calibrator = pickle.load(f)
        logger.info(f"Calibrator loaded (primary: {calibrator['primary']})")
    else:
        logger.warning("No calibrator found. Run: python src/calibrate.py")
        calibrator = None

    # Fusion layer (learned neural fusion: 11 features -> 1 probability)
    fusion_path = 'models/fusion_layer.pt'
    if os.path.exists(fusion_path):
        ckpt = torch.load(fusion_path, map_location=device, weights_only=False)
        fusion_model = FinalFusionLayer(
            input_dim=ckpt.get('input_dim', 11),
            hidden_dim=ckpt.get('hidden_dim', 64)
        ).to(device)
        fusion_model.load_state_dict(ckpt['fusion_state_dict'])
        fusion_model.eval()
        logger.info(f"Fusion layer loaded (Val AUC: {ckpt.get('val_auc', '?'):.4f})")
    else:
        logger.warning("No fusion layer found. Run: python train_fusion.py")
        fusion_model = None

    # Mock LLM (instant heuristic features for fusion layer)
    mock_llm = MockLLMAnalyzer()

    logger.info("All models loaded!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup"""
    load_models()
    yield
    logger.info("Shutting down...")


# ===================================================================
# FastAPI App
# ===================================================================
app = FastAPI(
    title="JobTrust API",
    description="Detect fraudulent job postings using AI (DistilBERT + Graph Features + Gemini LLM)",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow CORS (for browser extension / frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================================================================
# Request / Response Models
# ===================================================================
class JobPosting(BaseModel):
    """Input: Job posting data"""
    title: str = Field(..., description="Job title", examples=["Senior Software Engineer"])
    description: Optional[str] = Field(None, description="Job description")
    requirements: Optional[str] = Field(None, description="Job requirements")
    company_profile: Optional[str] = Field(None, description="Company description")
    benefits: Optional[str] = Field(None, description="Benefits offered")
    salary_range: Optional[str] = Field(None, description="Salary range")
    location: Optional[str] = Field(None, description="Job location")
    employment_type: Optional[str] = Field(None, description="Full-time, Part-time, etc.")
    contact_email: Optional[str] = Field(None, description="Contact email")
    department: Optional[str] = Field(None)
    has_company_logo: Optional[int] = Field(0)
    has_questions: Optional[int] = Field(0)
    telecommuting: Optional[int] = Field(0)
    required_experience: Optional[str] = Field(None)
    required_education: Optional[str] = Field(None)
    industry: Optional[str] = Field(None)
    function: Optional[str] = Field(None)
    source_url: Optional[str] = Field(None, description="URL of the page the job was scraped from")
    use_llm: Optional[bool] = Field(True, description="Use Gemini LLM for uncertain cases")


class GraphFeatures(BaseModel):
    domain_trust: float
    domain_age_days: float
    registrar: str
    email_free: bool
    has_linkedin: bool
    suspicious_domain: bool


class LLMResult(BaseModel):
    provider: str
    probability: float
    confidence: str
    red_flags: List[str]
    reasoning: str
    cost_usd: float
    cached: bool


class PredictionResponse(BaseModel):
    """Output: Prediction result"""
    fraud_probability: float = Field(..., description="Final fraud probability (0-1)")
    is_fake: bool = Field(..., description="Whether the job is likely fake")
    risk_level: str = Field(..., description="LOW / MEDIUM / HIGH")
    method: str = Field(..., description="model_only or hybrid (model + LLM)")
    model_probability: float = Field(..., description="Phase 2 model prediction")
    graph_features: GraphFeatures
    llm_result: Optional[LLMResult] = None
    explanation: str = Field(..., description="Human-readable explanation")
    processing_time_ms: int = Field(..., description="Total processing time in ms")


# ===================================================================
# Calibration helper
# ===================================================================
# Optimal raw threshold from evaluation (maximizes F1 without calibration)
RAW_OPTIMAL_THRESHOLD = 0.03

def calibrate_prob(raw_prob: float) -> float:
    """Apply calibration to a raw model probability."""
    if calibrator is None:
        return raw_prob
    
    method = calibrator['primary']  # 'isotonic' or 'platt'
    if method == 'isotonic':
        return float(calibrator['isotonic'].predict([raw_prob])[0])
    else:
        return float(calibrator['platt'].predict_proba([[raw_prob]])[0][1])


# ===================================================================
# Prediction Logic
# ===================================================================
def run_prediction(job: JobPosting) -> PredictionResponse:
    """Run the full hybrid prediction pipeline"""
    start = time.time()
    job_dict = job.model_dump()

    # --- Phase 2 Model ---
    text = f"Title: {job_dict.get('title', '')}. "
    text += f"Description: {job_dict.get('description', '')}. "
    text += f"Requirements: {job_dict.get('requirements', '')}. "
    text += f"Company: {job_dict.get('company_profile', '')}. "
    text += f"Benefits: {job_dict.get('benefits', '')}"

    encoding = tokenizer(text, max_length=512, padding='max_length',
                         truncation=True, return_tensors='pt')

    meta_vector = extract_metadata_from_job(job_dict, feature_names)
    meta_scaled = scaler.transform([meta_vector])[0]
    meta_tensor = torch.tensor(np.array([meta_scaled]), dtype=torch.float32).to(device)

    graph_obj = net_extractor.extract(
        company_profile=job_dict.get('company_profile', ''),
        contact_email=job_dict.get('contact_email'),
        company_name=job_dict.get('title', ''),
        source_url=job_dict.get('source_url'),
    )
    graph_tensor = torch.tensor([graph_obj.to_vector()], dtype=torch.float32).to(device)

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        raw_prob = model.predict_proba(input_ids, attention_mask,
                                        meta_tensor, graph_tensor).item()

    # Apply calibration
    cal_prob = calibrate_prob(raw_prob)

    # --- Build 11-dim fusion feature vector ---
    # Mock LLM gives instant heuristic features (no API call)
    mock_result = mock_llm.analyze(job_dict, raw_prob)
    llm_conf_map = {'high': 1.0, 'medium': 0.5, 'low': 0.25}

    gf_vec = graph_obj.to_vector()
    fusion_features = [
        raw_prob,                                                      # model_probability
        abs(raw_prob - 0.5) * 2,                                       # model_confidence
        mock_result.fraud_probability,                                 # llm_probability (mock)
        llm_conf_map.get(mock_result.confidence, 0.5),                 # llm_confidence
        min(len(mock_result.red_flags), 5) / 5.0,                     # llm_num_red_flags
        min(1.0, len(mock_result.reasoning) / 500),                    # llm_reasoning_score
        graph_obj.domain_trust_score,                                  # graph_domain_trust
        1.0 if gf_vec[3] else (0.0 if gf_vec[4] else 0.5),           # graph_email_type
        float(graph_obj.has_linkedin_page),                            # graph_linkedin
        float(graph_obj.is_known_fake_domain),                         # graph_suspicious
        sum([bool(job_dict.get(k)) for k in                            # metadata_richness
             ['description','requirements','company_profile','salary_range','location']]) / 5.0,
    ]

    # --- Fusion Layer (learned neural combination) ---
    final_prob = cal_prob
    method = "model_only"
    llm_result_data = None
    explanation = ""

    if fusion_model is not None:
        with torch.no_grad():
            fx = torch.tensor([fusion_features], dtype=torch.float32).to(device)
            fusion_logit = fusion_model(fx)
            fusion_prob = torch.sigmoid(fusion_logit).item()
        final_prob = fusion_prob
        method = "fusion (learned)"
        explanation = f"Fusion layer: {fusion_prob:.1%} (model raw={raw_prob:.1%}, mock_llm={mock_result.fraud_probability:.1%})"
    else:
        final_prob = cal_prob
        explanation = f"Calibrated model: {cal_prob:.1%} (raw={raw_prob:.1%})"

    # --- Real LLM (Gemini) for uncertain fusion outputs ---
    use_llm = job.use_llm and llm_analyzer is not None
    llm_trigger = 0.15 <= final_prob <= 0.85  # Fusion layer uncertain

    if use_llm and llm_trigger:
        try:
            llm_res = llm_analyzer.analyze(job_dict, final_prob)

            # Blend fusion + real LLM
            if llm_res.confidence == "high":
                final_prob = 0.3 * final_prob + 0.7 * llm_res.fraud_probability
            elif llm_res.confidence == "medium":
                final_prob = 0.5 * final_prob + 0.5 * llm_res.fraud_probability
            else:
                final_prob = 0.7 * final_prob + 0.3 * llm_res.fraud_probability

            method = "fusion + Gemini LLM"
            llm_result_data = LLMResult(
                provider=llm_res.provider,
                probability=llm_res.fraud_probability,
                confidence=llm_res.confidence,
                red_flags=llm_res.red_flags[:5],
                reasoning=llm_res.reasoning[:500],
                cost_usd=llm_res.cost_usd,
                cached=llm_res.cached,
            )
            explanation = (
                f"Fusion: {fusion_prob:.1%}, "
                f"Gemini ({llm_res.confidence}): {llm_res.fraud_probability:.1%}. "
                f"Final: {final_prob:.1%}. "
                f"Issues: {', '.join(llm_res.red_flags[:3]) if llm_res.red_flags else 'None'}"
            )
        except Exception as e:
            logger.error(f"LLM failed: {e}")

    # Risk level
    if final_prob < 0.3:
        risk = "LOW"
    elif final_prob < 0.7:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    elapsed = int((time.time() - start) * 1000)

    return PredictionResponse(
        fraud_probability=round(final_prob, 4),
        is_fake=final_prob > 0.5,
        risk_level=risk,
        method=method,
        model_probability=round(raw_prob, 4),
        graph_features=GraphFeatures(
            domain_trust=round(graph_obj.domain_trust_score, 2),
            domain_age_days=round(graph_obj.whois_domain_age_days, 0),
            registrar=graph_obj.whois_registrar or "Unknown",
            email_free=bool(graph_obj.email_is_free_provider),
            has_linkedin=bool(graph_obj.has_linkedin_page),
            suspicious_domain=bool(graph_obj.is_known_fake_domain),
        ),
        llm_result=llm_result_data,
        explanation=explanation,
        processing_time_ms=elapsed,
    )


# ===================================================================
# API Endpoints
# ===================================================================
@app.get("/")
def root():
    """API info"""
    return {
        "name": "JobTrust API",
        "version": "1.0.0",
        "model": "Phase 2 (DistilBERT + Metadata + Graph + LLM)",
        "llm": llm_analyzer.provider if llm_analyzer else "disabled",
        "endpoints": {
            "POST /predict": "Analyze a job posting",
            "GET /health": "Health check",
        }
    }


@app.get("/health")
def health():
    """Health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "llm_available": llm_analyzer is not None,
        "llm_provider": llm_analyzer.provider if llm_analyzer else None,
        "device": str(device),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(job: JobPosting):
    """
    Analyze a job posting for fraud.
    
    Returns fraud probability, risk level, and detailed analysis.
    If the model is uncertain, Gemini LLM provides additional analysis (free).
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    return run_prediction(job)


# ===================================================================
# Run
# ===================================================================
if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("  JOBTRUST API")
    print("  http://localhost:8000")
    print("  Docs: http://localhost:8000/docs")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
