# Phase 3: LLM + Fusion Layer

Adds LLM analysis and a learned fusion layer on top of the Phase 2 model.

## How It Works

```
Phase 2 Model (raw prob) ---+
Mock LLM (heuristics)    ---+--> 11-dim features --> Fusion Layer --> Final Prediction
Graph features            ---+
Metadata richness         ---+

If fusion output is uncertain (0.15-0.85):
  --> Gemini LLM (free) provides deep analysis with red flags
```

## Files

| File | Purpose |
|------|---------|
| `llm_analyzer_v2.py` | Multi-provider LLM (Gemini/OpenAI/Mock) |
| `mock_llm.py` | Heuristic-based mock LLM (instant, no API) |
| `final_fusion_model.py` | Learned neural fusion layer (11 -> 64 -> 32 -> 1) |
| `hybrid_predictor_complete.py` | Complete hybrid pipeline |
| `feature_utils.py` | Shared metadata extraction utility |
| `config.py` | Phase 3 configuration (API keys, thresholds) |

## Usage

```python
from phase3.llm_analyzer_v2 import LLMJobAnalyzer

analyzer = LLMJobAnalyzer(provider="auto")  # auto-selects Gemini > OpenAI > Mock

result = analyzer.analyze({
    'title': 'Work From Home - $500/Day!',
    'description': 'Send $50 for starter kit.',
})

print(f"Fraud: {result.fraud_probability:.1%}")
print(f"Red flags: {result.red_flags}")
```

## LLM Providers

| Provider | Cost | Setup |
|----------|------|-------|
| Gemini | FREE | `GEMINI_API_KEY` in `.env` |
| OpenAI | ~$0.005/call | `OPENAI_API_KEY` in `.env` |
| Mock | FREE | No setup needed |
