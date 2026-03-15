"""
LLM Analyzer with Multi-Provider Support
Supports: OpenAI, Gemini (Google), and Mock
"""
import os
import json
import hashlib
import pickle
import time
import logging
from typing import Dict, Optional
from dataclasses import dataclass

import numpy as np

# Try to load .env file for API keys
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional

logger = logging.getLogger(__name__)

# Try to import providers
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try new google-genai SDK first, fall back to deprecated google-generativeai
GEMINI_AVAILABLE = False
GEMINI_NEW_SDK = False

try:
    from google import genai as genai_new
    GEMINI_AVAILABLE = True
    GEMINI_NEW_SDK = True
except ImportError:
    try:
        import google.generativeai as genai_old
        GEMINI_AVAILABLE = True
        GEMINI_NEW_SDK = False
    except ImportError:
        pass


@dataclass
class LLMAnalysis:
    """Result from LLM analysis"""
    fraud_probability: float
    confidence: str  # "high", "medium", "low"
    reasoning: str
    red_flags: list
    recommendations: list
    cost_usd: float
    cached: bool = False
    provider: str = "unknown"


class LLMJobAnalyzer:
    """
    Multi-provider LLM analyzer
    Supports OpenAI, Gemini (free tier), and Mock
    """
    
    def __init__(self, 
                 provider: str = "auto",  # "openai", "gemini", "mock", or "auto"
                 api_key: Optional[str] = None):
        """
        Args:
            provider: LLM provider to use
            api_key: API key (if None, reads from env var)
        """
        self.provider = provider
        self.client = None
        self.gemini_client = None
        self.gemini_model_name = None
        
        # Auto-select provider
        if provider == "auto":
            self.provider = self._auto_select_provider()
        
        # Initialize selected provider
        if self.provider == "openai":
            self._init_openai(api_key)
        elif self.provider == "gemini":
            self._init_gemini(api_key)
        elif self.provider == "mock":
            print("Using Mock LLM (no API calls)")
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        
        # Cache setup
        self.cache_dir = "processed/cache/llm"
        os.makedirs(self.cache_dir, exist_ok=True)
        self._cache = self._load_cache()
        
        print(f"LLM Analyzer ready (Provider: {self.provider})")
    
    def _auto_select_provider(self) -> str:
        """Automatically select best available provider"""
        # Priority: Gemini (free) > OpenAI > Mock
        
        if GEMINI_AVAILABLE and os.getenv('GEMINI_API_KEY'):
            print("Auto-selected: Gemini (Google)")
            return "gemini"
        
        if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
            print("Auto-selected: OpenAI")
            return "openai"
        
        print("Auto-selected: Mock (no API keys found)")
        return "mock"
    
    def _init_openai(self, api_key: Optional[str]):
        """Initialize OpenAI client"""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not installed. Run: pip install openai")
        
        key = api_key or os.getenv('OPENAI_API_KEY')
        if not key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var.")
        
        self.client = OpenAI(api_key=key)
        self.model = "gpt-4o-mini"
        self.cost_per_call = 0.005
        print(f"OpenAI initialized ({self.model})")
    
    def _init_gemini(self, api_key: Optional[str]):
        """Initialize Gemini (Google) client using new or old SDK"""
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Gemini not installed. Run: pip install google-genai"
            )
        
        key = api_key or os.getenv('GEMINI_API_KEY')
        if not key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY env var or pass api_key=."
            )
        
        # Models to try in order of preference (newest first)
        model_candidates = [
            'gemini-2.5-flash',       # Latest and fastest (free tier)
            'gemini-2.0-flash',       # Previous gen fast model (free tier)
            'gemini-2.0-flash-lite',  # Lightweight alternative
        ]
        
        if GEMINI_NEW_SDK:
            self._init_gemini_new_sdk(key, model_candidates)
        else:
            self._init_gemini_old_sdk(key, model_candidates)
    
    def _init_gemini_new_sdk(self, key: str, model_candidates: list):
        """Initialize using new google-genai SDK"""
        self.gemini_client = genai_new.Client(api_key=key)
        
        last_error = None
        for model_name in model_candidates:
            try:
                # Test with a minimal call
                response = self.gemini_client.models.generate_content(
                    model=model_name,
                    contents="Say OK"
                )
                _ = response.text  # Verify we got a response
                self.gemini_model_name = model_name
                self.cost_per_call = 0.0
                print(f"Gemini initialized ({model_name} - FREE TIER, new SDK)")
                return
            except Exception as e:
                last_error = e
                logger.debug(f"Model {model_name} failed: {e}")
                continue
        
        raise RuntimeError(
            f"Failed to initialize any Gemini model. Last error: {last_error}\n"
            f"Tried models: {model_candidates}\n"
            f"Possible causes:\n"
            f"  1. Invalid API key (check GEMINI_API_KEY)\n"
            f"  2. API key not enabled for Gemini API\n"
            f"  3. Network/firewall issue\n"
            f"  Get a key at: https://aistudio.google.com/app/apikey"
        )
    
    def _init_gemini_old_sdk(self, key: str, model_candidates: list):
        """Initialize using deprecated google-generativeai SDK"""
        genai_old.configure(api_key=key)
        
        last_error = None
        for model_name in model_candidates:
            try:
                model = genai_old.GenerativeModel(model_name)
                model.generate_content("Say OK")
                self.gemini_client = model  # Store the model object
                self.gemini_model_name = model_name
                self.cost_per_call = 0.0
                print(f"Gemini initialized ({model_name} - FREE TIER, legacy SDK)")
                return
            except Exception as e:
                last_error = e
                logger.debug(f"Model {model_name} failed: {e}")
                continue
        
        raise RuntimeError(
            f"Failed to initialize any Gemini model. Last error: {last_error}\n"
            f"Tried models: {model_candidates}\n"
            f"Get a key at: https://aistudio.google.com/app/apikey"
        )
    
    def _get_cache_key(self, job_text: str) -> str:
        """Generate cache key from job text"""
        return hashlib.md5(job_text.encode()).hexdigest()[:16]
    
    def _load_cache(self) -> Dict:
        """Load LLM response cache"""
        cache_file = os.path.join(self.cache_dir, f"{self.provider}_responses.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except (pickle.UnpicklingError, EOFError, FileNotFoundError) as e:
                logger.warning(f"Failed to load cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to disk"""
        cache_file = os.path.join(self.cache_dir, f"{self.provider}_responses.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(self._cache, f)
    
    def _check_cache(self, cache_key: str) -> Optional[LLMAnalysis]:
        """Check if response is cached"""
        if cache_key in self._cache:
            result, timestamp = self._cache[cache_key]
            # 24 hour TTL
            if time.time() - timestamp < 24 * 3600:
                result.cached = True
                return result
        return None
    
    def _create_prompt(self, job_dict: Dict) -> str:
        """Create detailed prompt for LLM"""
        # Safely get string values (handle None)
        def safe(key, default='N/A', maxlen=None):
            val = job_dict.get(key)
            val = str(val) if val else default
            return val[:maxlen] if maxlen else val

        return f"""Analyze this job posting for fraud detection.

JOB TITLE: {safe('title')}

COMPANY: {safe('company_profile', maxlen=500)}

DESCRIPTION: {safe('description', maxlen=1000)}

REQUIREMENTS: {safe('requirements', maxlen=500)}

SALARY: {safe('salary_range', 'Not mentioned')}
LOCATION: {safe('location', 'Not mentioned')}

Check for:
1. Advance fee fraud (upfront payment requests)
2. Identity theft (excessive personal info)
3. MLM/Pyramid schemes
4. Unrealistic pay for experience
5. Vague descriptions with high pay
6. Urgency pressure tactics

Respond in JSON format:
{{
    "fraud_probability": 0.0-1.0,
    "confidence": "high|medium|low",
    "reasoning": "detailed analysis",
    "red_flags": ["flag1", "flag2"],
    "recommendations": ["rec1", "rec2"]
}}"""
    
    def _parse_response(self, content: str) -> Dict:
        """Parse LLM response"""
        # Clean up markdown
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        try:
            return json.loads(content)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}")
            # Fallback parsing
            return {
                "fraud_probability": 0.5,
                "confidence": "low",
                "reasoning": f"Failed to parse LLM response: {content[:200]}",
                "red_flags": [],
                "recommendations": ["Manual review needed"]
            }
    
    def analyze(self, job_dict: Dict, model_prob: Optional[float] = None) -> LLMAnalysis:
        """
        Analyze job using selected LLM provider
        """
        prompt = self._create_prompt(job_dict)
        cache_key = self._get_cache_key(prompt)
        
        # Check cache
        cached = self._check_cache(cache_key)
        if cached:
            return cached
        
        # Call appropriate provider
        if self.provider == "openai":
            result = self._call_openai(prompt)
        elif self.provider == "gemini":
            result = self._call_gemini(prompt)
        elif self.provider == "mock":
            result = self._call_mock(job_dict)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        
        # Cache and return
        result.provider = self.provider
        self._cache[cache_key] = (result, time.time())
        self._save_cache()
        
        return result
    
    def _call_openai(self, prompt: str) -> LLMAnalysis:
        """Call OpenAI API"""
        if not self.client:
            raise Exception("OpenAI client not initialized")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a fraud detection expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.1
        )
        
        content = response.choices[0].message.content
        parsed = self._parse_response(content)
        
        return LLMAnalysis(
            fraud_probability=parsed.get('fraud_probability', 0.5),
            confidence=parsed.get('confidence', 'medium'),
            reasoning=parsed.get('reasoning', 'No reasoning'),
            red_flags=parsed.get('red_flags', []),
            recommendations=parsed.get('recommendations', []),
            cost_usd=self.cost_per_call,
            cached=False,
            provider="openai"
        )
    
    def _call_gemini(self, prompt: str) -> LLMAnalysis:
        """Call Gemini (Google) API using new or old SDK"""
        if not self.gemini_client:
            raise RuntimeError("Gemini client not initialized")
        
        # Add JSON instruction
        full_prompt = prompt + "\n\nIMPORTANT: Respond ONLY with valid JSON, no markdown, no explanation outside JSON."
        
        try:
            # Call using appropriate SDK
            if GEMINI_NEW_SDK:
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model_name,
                    contents=full_prompt
                )
                content = response.text if response and response.text else None
                if not content:
                    logger.warning("Gemini returned empty response")
                    return LLMAnalysis(
                        fraud_probability=0.5, confidence="low",
                        reasoning="Gemini returned empty response. Manual review needed.",
                        red_flags=[], recommendations=["Manual review recommended"],
                        cost_usd=0.0, cached=False, provider="gemini"
                    )
            else:
                # Old SDK -- gemini_client is a GenerativeModel instance
                response = self.gemini_client.generate_content(full_prompt)
                if not response.parts:
                    logger.warning("Gemini returned empty response (possibly blocked)")
                    return LLMAnalysis(
                        fraud_probability=0.5, confidence="low",
                        reasoning="Gemini response blocked by safety filters.",
                        red_flags=[], recommendations=["Manual review recommended"],
                        cost_usd=0.0, cached=False, provider="gemini"
                    )
                content = response.text
            
            parsed = self._parse_response(content)
            
            return LLMAnalysis(
                fraud_probability=parsed.get('fraud_probability', 0.5),
                confidence=parsed.get('confidence', 'medium'),
                reasoning=parsed.get('reasoning', 'No reasoning'),
                red_flags=parsed.get('red_flags', []),
                recommendations=parsed.get('recommendations', []),
                cost_usd=0.0,  # FREE!
                cached=False,
                provider="gemini"
            )
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return LLMAnalysis(
                fraud_probability=0.5,
                confidence="low",
                reasoning=f"Gemini API error: {str(e)}",
                red_flags=[],
                recommendations=["Manual review recommended - API error"],
                cost_usd=0.0,
                cached=False,
                provider="gemini"
            )
    
    def _call_mock(self, job_dict: Dict) -> LLMAnalysis:
        """Mock LLM for testing (no API call)"""
        from .mock_llm import MockLLMAnalyzer
        mock = MockLLMAnalyzer()
        return mock.analyze(job_dict)


# Example usage
if __name__ == "__main__":
    # Test with Gemini (free!)
    # Set: GEMINI_API_KEY=your-key
    
    analyzer = LLMJobAnalyzer(provider="gemini")  # or "openai", "mock", "auto"
    
    job = {
        'title': 'Work From Home - $500/Day!',
        'description': 'No experience needed! Send $50 for starter kit.',
        'company_profile': 'Quick money!'
    }
    
    result = analyzer.analyze(job)
    print(f"Provider: {result.provider}")
    print(f"Fraud: {result.fraud_probability:.1%}")
    print(f"Cost: ${result.cost_usd}")
