# Fake Job Detector

AI-powered system to detect fraudulent job postings using a multi-modal deep learning pipeline with LLM verification.

Trained on the [Real/Fake Job Postings dataset](https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction) (17,880 job postings, 866 fraudulent).

## Architecture

```
Job Posting
    |
    v
+---------------------------+
| Phase 2 Model             |
| DistilBERT (768-dim text) |
| + Metadata (41 features)  |
| + Graph (14 features)     |
+---------------------------+
    |  raw probability
    v
+---------------------------+
| Fusion Layer (learned)    |
| 11-dim features:          |
|  - model prob + confidence|
|  - mock LLM heuristics    |
|  - graph trust signals    |
|  - metadata richness      |
| Neural Net: 11->64->32->1 |
+---------------------------+
    |  fusion probability
    v
+---------------------------+
| Gemini LLM (if uncertain) |
| FREE via Google API       |
| Only called for 0.15-0.85 |
+---------------------------+
    |
    v
  Final Prediction
  (probability, risk level, explanation)
```

## Performance

| Metric | Phase 2 Model | Fusion Layer | Fusion + Gemini |
|--------|---------------|--------------|-----------------|
| AUC    | 0.9684        | 0.9568       | -               |
| F1     | 0.8035        | 0.8100       | -               |

### Example Predictions

| Job Posting | Raw Model | Fusion | Fusion + Gemini |
|-------------|-----------|--------|-----------------|
| Scam ($500/day, send $50) | 15.7% | **99.8%** | 99.8% |
| Google SWE (legit) | 2.3% | **0.0%** | 0.0% |
| MLM scam ($200 starter kit) | 3.9% | **94.7%** | 94.7% |

## Project Structure

```
prediction fake/
|
|-- api.py                      # FastAPI REST API (main entry point)
|-- train_fusion.py             # Train the neural fusion layer
|-- evaluate_full.py            # Full evaluation on test set
|-- requirements.txt            # Python dependencies
|-- .env                        # API keys (not committed)
|-- .env.example                # Template for API keys
|
|-- src/                        # Core source code
|   |-- data_processor.py       # Data cleaning, feature extraction, splits
|   |-- model.py                # Phase 1: DistilBERT + Metadata model
|   |-- dataset.py              # PyTorch Dataset (Phase 1)
|   |-- train.py                # Phase 1 training
|   |-- predict.py              # Phase 1 inference
|   |-- evaluate.py             # Phase 1 evaluation
|   |-- calibrate.py            # Platt/Isotonic calibration
|   |-- focal_loss.py           # Focal loss for imbalanced data
|   |
|   |-- phase2/                 # Phase 2: + Graph/network features
|   |   |-- model_v2.py         # DistilBERT + Metadata + Graph model
|   |   |-- dataset_v2.py       # PyTorch Dataset with graph features
|   |   |-- network_features.py # Graph feature extraction (domain, email, etc.)
|   |   |-- train_phase2.py     # Training (supports --loss focal)
|   |   |-- data_enrichment.py  # Enrich Phase 1 data with graph features
|   |   +-- config.py           # Phase 2 configuration
|   |
|   +-- phase3/                 # Phase 3: LLM + Fusion layer
|       |-- llm_analyzer_v2.py  # Multi-provider LLM (Gemini/OpenAI/Mock)
|       |-- mock_llm.py         # Heuristic-based mock LLM (instant, free)
|       |-- final_fusion_model.py # Learned neural fusion layer (11->1)
|       |-- feature_utils.py    # Shared metadata extraction
|       |-- config.py           # Phase 3 configuration
|       +-- hybrid_predictor_complete.py # Complete hybrid pipeline
|
|-- extension/                  # Chrome browser extension (Manifest V3)
|   |-- manifest.json           # Extension config
|   |-- background.js           # Service worker
|   |-- content.js              # Content script (LinkedIn, Indeed, etc.)
|   |-- popup.html/js/css       # Extension popup UI
|   +-- options.html            # Extension settings page
|
|-- tests/                      # Test scripts
|   |-- test_api.py             # API endpoint tests
|   |-- test_gemini.py          # Gemini LLM integration test
|   |-- test_hybrid_pipeline.py # Full pipeline test (model + LLM)
|   +-- test_phase2.py          # Phase 2 model test
|
|-- models/                     # Trained models (not committed, ~256MB)
|   |-- phase2_best.pt          # Phase 2 model checkpoint
|   |-- calibrator.pkl          # Isotonic/Platt calibrator
|   |-- fusion_layer.pt         # Trained fusion layer
|   +-- eval_plots/             # ROC curve, confusion matrix, etc.
|
|-- processed/                  # Preprocessed data (not committed)
|   |-- train_v2.pkl, val_v2.pkl, test_v2.pkl
|   |-- scaler.pkl, feature_names.pkl
|   +-- fusion_features.pkl
|
+-- notebooks/                  # Google Colab training notebooks
    |-- train_colab_full.ipynb  # Phase 1 full training
    +-- Phase2_Colab_Final.ipynb # Phase 2 training
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get a Gemini API Key (FREE)

1. Go to https://aistudio.google.com/app/apikey
2. Create an API key
3. Copy `.env.example` to `.env` and paste your key:

```bash
copy .env.example .env
```

Edit `.env`:
```
GEMINI_API_KEY=your-key-here
```

### 3. Preprocess Data (if starting from scratch)

```bash
cd src
python data_processor.py
cd phase2
python data_enrichment.py
```

### 4. Train Models (if starting from scratch)

**Phase 1** (DistilBERT + Metadata):
```bash
cd src
python train.py
```

**Phase 2** (+ Graph features):
```bash
cd src/phase2
python train_phase2.py --loss focal
```

**Calibration** (no retraining needed):
```bash
python src/calibrate.py
```

**Fusion Layer** (uses Mock LLM, no API cost):
```bash
python train_fusion.py
```

### 5. Run the API

```bash
python api.py
```

The API starts at **http://localhost:8000**. Interactive docs at **http://localhost:8000/docs**.

### 6. Load the Browser Extension

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode" (top right)
3. Click "Load unpacked" and select the `extension/` folder
4. Navigate to any job listing on LinkedIn, Indeed, Glassdoor, etc.

## API Usage

### Analyze a Job Posting

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Work From Home - $500/Day!",
    "description": "No experience needed! Send $50 for starter kit.",
    "company_profile": "Quick money!",
    "contact_email": "scam@gmail.com"
  }'
```

### Response

```json
{
  "fraud_probability": 0.998,
  "is_fake": true,
  "risk_level": "HIGH",
  "method": "fusion (learned)",
  "model_probability": 0.157,
  "graph_features": {
    "domain_trust": 0.4,
    "email_free": true,
    "has_linkedin": false,
    "suspicious_domain": false
  },
  "explanation": "Fusion layer: 99.8% (model raw=15.7%, mock_llm=95.0%)",
  "processing_time_ms": 160
}
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| POST | `/predict` | Analyze a job posting |
| GET | `/docs` | Interactive Swagger UI |

## How It Works

1. **Text Encoding**: DistilBERT converts the job description into a 768-dimensional embedding
2. **Metadata**: 41 features extracted (text lengths, salary presence, education level, etc.)
3. **Graph Features**: 14 network signals (domain trust, email type, LinkedIn presence, etc.)
4. **Phase 2 Model**: Fuses all three into a raw fraud probability
5. **Fusion Layer**: A trained neural network combines the model output with heuristic LLM signals and graph features for a calibrated prediction
6. **Gemini LLM** (optional): For uncertain cases (0.15-0.85), the free Gemini API provides deep analysis with red flags and reasoning

## Tech Stack

- **PyTorch** + **Transformers** (DistilBERT)
- **scikit-learn** (calibration, metrics)
- **FastAPI** + **Uvicorn** (REST API)
- **Google Gemini API** (free LLM)
- **Chrome Extension** (Manifest V3)

## License

This project is for educational and research purposes.
