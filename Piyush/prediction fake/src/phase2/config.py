"""
Centralized configuration for Phase 2
Follows 12-factor app principles - configuration via environment variables
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional
import json


@dataclass
class Phase2Config:
    """
    Configuration class for Phase 2 features.
    
    Attributes:
        graph_feature_dim: Dimension of graph features (output)
        domain_cache_ttl: Cache TTL for domain lookups in hours
        enable_linkedin: Whether to use LinkedIn API
        enable_whois: Whether to use WHOIS lookups
        max_workers: Max parallel workers for enrichment
        cache_dir: Directory for caching API responses
    """
    
    # Feature dimensions
    graph_feature_dim: int = 32
    metadata_dim: int = 41  # From Phase 1
    text_embedding_dim: int = 768  # DistilBERT
    
    # Combined dimension for fusion layer
    @property
    def total_input_dim(self) -> int:
        return self.text_embedding_dim + self.metadata_dim + self.graph_feature_dim
    
    # Network/Cache settings
    domain_cache_ttl: int = 24  # hours
    request_timeout: int = 10  # seconds
    max_retries: int = 3
    max_workers: int = 5
    
    # Feature flags (enable/disable features)
    enable_whois: bool = True
    enable_linkedin: bool = False  # Requires API key
    enable_glassdoor: bool = False  # Requires API key
    
    # Paths
    cache_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        '..', 'processed', 'cache'
    ))
    
    # Model architecture
    fusion_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    dropout_rate: float = 0.4
    
    # Training
    learning_rate_bert: float = 2e-5
    learning_rate_other: float = 2e-4
    batch_size: int = 16
    epochs: int = 10
    patience: int = 3
    
    # API Keys (load from env)
    linkedin_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv('LINKEDIN_API_KEY')
    )
    glassdoor_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv('GLASSDOOR_API_KEY')
    )
    whois_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv('WHOIS_API_KEY')
    )
    
    def __post_init__(self):
        """Validate configuration"""
        os.makedirs(self.cache_dir, exist_ok=True)
        
        if self.enable_linkedin and not self.linkedin_api_key:
            print("Warning: LinkedIn enabled but no API key found. Set LINKEDIN_API_KEY env var.")
            self.enable_linkedin = False
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            'graph_feature_dim': self.graph_feature_dim,
            'total_input_dim': self.total_input_dim,
            'enable_whois': self.enable_whois,
            'enable_linkedin': self.enable_linkedin,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'dropout_rate': self.dropout_rate,
        }
    
    def save(self, path: str):
        """Save config to JSON"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Phase2Config':
        """Load config from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# Global config instance
_config: Optional[Phase2Config] = None


def get_config() -> Phase2Config:
    """Get or create global config instance (Singleton pattern)"""
    global _config
    if _config is None:
        _config = Phase2Config()
    return _config


def set_config(config: Phase2Config):
    """Set global config instance"""
    global _config
    _config = config
