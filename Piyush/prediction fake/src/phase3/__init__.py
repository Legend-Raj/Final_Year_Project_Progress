"""
Phase 3: LLM Integration for Edge Cases
Hybrid approach: Model + LLM for uncertain predictions
"""

from .llm_analyzer_v2 import LLMJobAnalyzer, LLMAnalysis
from .mock_llm import MockLLMAnalyzer
from .config import Phase3Config

__all__ = ['LLMJobAnalyzer', 'LLMAnalysis', 'MockLLMAnalyzer', 'Phase3Config']
