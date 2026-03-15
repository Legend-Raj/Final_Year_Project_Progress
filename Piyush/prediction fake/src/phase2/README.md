# Phase 2: Graph/Network Features

Phase 2 extends Phase 1 by adding **network-based features** for improved fake job detection.

## 🏗️ Architecture

```
Phase 2 Model:
┌─────────────────────────────────────────────────────────┐
│  INPUT LAYERS                                           │
├─────────────────────────────────────────────────────────┤
│  DistilBERT (Text)  ───────┐  768-dim                  │
│  Metadata (Tabular) ───────┼──┐  41-dim                │
│  Graph Features     ───────┼──┼──┐  14-dim → 32-dim    │
│                            ▼  ▼  ▼                      │
│  YOUR FUSION LAYERS (512 → 256 → 128)                   │
│                            │                            │
│                            ▼                            │
│                      OUTPUT (1-dim)                     │
└─────────────────────────────────────────────────────────┘
```

## 🌐 Graph Features

| Feature | Description | Range |
|---------|-------------|-------|
| `domain_age_years` | Domain age | 0-∞ |
| `domain_has_ssl` | SSL certificate | 0/1 |
| `domain_suspicious_tld` | Suspicious TLD (.tk, .ml, etc.) | 0/1 |
| `email_is_corporate` | Corporate email domain | 0/1 |
| `email_is_free_provider` | Gmail/Yahoo/etc. | 0/1 |
| `email_domain_age_years` | Email domain age | 0-∞ |
| `has_linkedin_page` | LinkedIn presence | 0/1 |
| `linkedin_employee_count_log` | Employee count (log) | 0-∞ |
| `linkedin_followers_log` | Followers (log) | 0-∞ |
| `website_has_contact_page` | Contact page present | 0/1 |
| `website_has_career_page` | Career page present | 0/1 |
| `website_professional_score` | Website quality | 0-1 |
| `domain_trust_score` | Overall trust score | 0-1 |
| `is_known_fake_domain` | Known fake pattern | 0/1 |

## 🚀 Quick Start

### 1. Enrich Phase 1 Data

```bash
cd src/phase2
python data_enrichment.py --processed-dir ../../processed
```

This creates:
- `processed/train_v2.pkl`
- `processed/val_v2.pkl`
- `processed/test_v2.pkl`

### 2. Train Phase 2 Model

```bash
python train_phase2.py \
    --use-phase1 ../../models/best_model_full.pt \
    --batch-size 16 \
    --epochs 10
```

### 3. Make Predictions

```python
from phase2.predict_phase2 import FakeJobPredictorV2

predictor = FakeJobPredictorV2()
result = predictor.predict(job_dict, return_features=True)

print(f"Fraud Probability: {result['fraud_probability']:.2%}")
print(f"Risk Level: {result['risk_level']}")

# Get explanation
print(predictor.explain_prediction(job_dict))
```

## 📁 File Structure

```
phase2/
├── __init__.py              # Package initialization
├── config.py                # Central configuration
├── network_features.py      # Feature extraction
├── graph_encoder.py         # Graph neural network (placeholder for future)
├── model_v2.py              # Phase 2 model
├── dataset_v2.py            # Dataset with graph features
├── data_enrichment.py       # Enrich Phase 1 data
├── train_phase2.py          # Training script
└── predict_phase2.py        # Inference script
```

## 🔧 Configuration

Modify `config.py` or use environment variables:

```python
from phase2.config import Phase2Config, set_config

config = Phase2Config(
    graph_feature_dim=32,
    enable_whois=True,
    enable_linkedin=False,  # Requires API key
    batch_size=16
)
set_config(config)
```

## 🔄 Backward Compatibility

Phase 2 is fully backward compatible with Phase 1:

```python
# Load Phase 1 weights into Phase 2 model
from phase2.model_v2 import create_model_v2_from_phase1

model = create_model_v2_from_phase1('models/best_model_full.pt')
```

The graph encoder will be randomly initialized and trained.

## 📊 Expected Improvements

| Metric | Phase 1 | Phase 2 (Expected) |
|--------|---------|-------------------|
| ROC-AUC | 0.963 | 0.970+ |
| F1 Score | 0.500 | 0.550+ |
| False Negative Rate | ~10% | ~5% |

## 🔮 Future Enhancements

1. **Real WHOIS API**: Integrate python-whois for actual domain age
2. **LinkedIn API**: Real company data via LinkedIn
3. **Graph Neural Networks**: Company relationship graphs
4. **Real-time Scraping**: Live website quality checks

## 🐛 Troubleshooting

### Import Error
```bash
# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:./src"
```

### CUDA Out of Memory
```python
# Reduce batch size in config
config = Phase2Config(batch_size=8)
```

### Cache Issues
```bash
# Clear feature cache
rm processed/cache/network_features_cache.pkl
```
