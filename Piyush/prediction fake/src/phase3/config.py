"""
Phase 3 Configuration
"""
import os
from dataclasses import dataclass
from typing import Optional

# Try to load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class Phase3Config:
    """Configuration for Phase 3 LLM integration"""
    
    # LLM Provider: "openai", "gemini", "mock", or "auto"
    llm_provider: str = "auto"
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Model settings
    llm_model: str = "gpt-4o-mini"  # For OpenAI provider
    gemini_model: str = "gemini-2.0-flash"  # For Gemini provider (FREE)
    
    # Cost control
    max_tokens: int = 500
    temperature: float = 0.1  # Low for consistent results
    
    # Thresholds for LLM triggering
    # Model's optimal raw threshold is ~0.03 (maximizes F1)
    # LLM is triggered for any suspicious prediction (above raw threshold)
    # that isn't already highly confident after calibration
    raw_optimal_threshold: float = 0.03  # Raw model threshold (from evaluation)
    uncertainty_low: float = 0.03        # Raw prob above this = potentially suspicious
    uncertainty_high: float = 0.85       # Calibrated prob above this = model confident enough
    
    # Caching
    cache_llm_responses: bool = True
    llm_cache_ttl: int = 24  # hours
    
    def __post_init__(self):
        # Load API keys from environment if not provided
        if not self.openai_api_key:
            self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.gemini_api_key:
            self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not self.anthropic_api_key:
            self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        
        # Auto-select provider based on available keys
        if self.llm_provider == "auto":
            if self.gemini_api_key:
                self.llm_provider = "gemini"
            elif self.openai_api_key:
                self.llm_provider = "openai"
            else:
                self.llm_provider = "mock"


# Cost estimates (per call)
COST_PER_CALL = {
    "gpt-4o-mini": 0.005,         # ~$0.005 per analysis
    "gpt-4o": 0.05,               # ~$0.05 per analysis
    "gemini-2.0-flash": 0.0,      # FREE tier
    "gemini-1.5-flash": 0.0,      # FREE tier
    "claude-3-haiku": 0.005,      # Similar to GPT-4o-mini
}
