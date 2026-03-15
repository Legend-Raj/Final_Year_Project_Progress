"""
Phase 2: Graph/Network Features Integration

This module extends Phase 1 by adding network-based features:
- Domain age and reputation
- Company LinkedIn presence
- Website quality metrics
- Email domain verification

Usage:
    from src.phase2.model_v2 import FakeJobDetectorV2
    from src.phase2.config import Phase2Config
"""

__version__ = "2.0.0"
__author__ = "Fake Job Detection Team"

from .config import Phase2Config
from .network_features import NetworkFeatureExtractor
from .model_v2 import FakeJobDetectorV2

__all__ = [
    "Phase2Config",
    "NetworkFeatureExtractor", 
    "FakeJobDetectorV2",
]
