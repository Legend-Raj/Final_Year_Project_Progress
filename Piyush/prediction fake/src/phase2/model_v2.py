"""
Phase 2 Model: DistilBERT + Metadata + Graph Features
Extends Phase 1 model with network-based features
"""
import torch
import torch.nn as nn
from transformers import DistilBertModel
from typing import Optional, Dict
import logging

from .config import Phase2Config, get_config

logger = logging.getLogger(__name__)


class GraphFeatureEncoder(nn.Module):
    """
    Encoder for graph/network features
    """
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(64, 48),
            nn.LayerNorm(48),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(48, output_dim),
            nn.ReLU()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class FakeJobDetectorV2(nn.Module):
    """
    Phase 2 Model: Multi-modal fusion with graph features
    
    Architecture:
    - Text: DistilBERT (768-dim)
    - Metadata: Tabular features (41-dim)
    - Graph: Network features (32-dim)
    - Fusion: Concatenate all → Deep layers
    
    Backward compatible: Can load Phase 1 weights for text/metadata parts
    """
    
    def __init__(self, 
                 num_meta_features: int = 41,
                 num_graph_features: int = 14,  # NetworkFeatures output
                 graph_embedding_dim: int = 32,
                 dropout_rate: float = 0.4,
                 freeze_bert_layers: int = 4):
        super().__init__()
        
        self.config = get_config()
        self.num_meta_features = num_meta_features
        self.num_graph_features = num_graph_features
        self.graph_embedding_dim = graph_embedding_dim
        
        # 1. Text Encoder (DistilBERT)
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.text_dropout = nn.Dropout(0.2)
        
        # Freeze bottom layers
        if freeze_bert_layers > 0:
            for param in self.distilbert.transformer.layer[:freeze_bert_layers].parameters():
                param.requires_grad = False
            logger.info(f"Frozen first {freeze_bert_layers} BERT layers")
        
        # 2. Metadata Encoder (from Phase 1)
        self.metadata_encoder = nn.Sequential(
            nn.Linear(num_meta_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 3. Graph Feature Encoder (NEW in Phase 2)
        self.graph_encoder = GraphFeatureEncoder(
            input_dim=num_graph_features,
            output_dim=graph_embedding_dim,
            dropout=0.3
        )
        
        # 4. Fusion Layers (Updated for Phase 2)
        # 768 (text) + 64 (meta) + 32 (graph) = 864
        combined_dim = 768 + 64 + graph_embedding_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate - 0.1),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 1)
        )
        
        self._init_weights()
        
        # Log model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Model V2 initialized: {total_params:,} total params, {trainable_params:,} trainable")
    
    def _init_weights(self):
        """Initialize custom layers"""
        for module in [self.metadata_encoder, self.graph_encoder, self.fusion]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                metadata: torch.Tensor,
                graph_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: [batch, seq_len] Text token IDs
            attention_mask: [batch, seq_len] Attention mask
            metadata: [batch, num_meta] Metadata features
            graph_features: [batch, num_graph] Graph features (optional)
            
        Returns:
            logits: [batch] Raw prediction scores
        """
        # Text encoding
        bert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = bert_output.last_hidden_state[:, 0, :]  # [CLS] token
        text_features = self.text_dropout(text_features)  # [batch, 768]
        
        # Metadata encoding
        meta_features = self.metadata_encoder(metadata)  # [batch, 64]
        
        # Graph encoding (NEW)
        if graph_features is not None:
            graph_encoded = self.graph_encoder(graph_features)  # [batch, 32]
        else:
            # If no graph features, use zeros (backward compatibility)
            batch_size = input_ids.size(0)
            graph_encoded = torch.zeros(batch_size, self.graph_embedding_dim,
                                       device=input_ids.device, dtype=torch.float32)
        
        # Fusion
        combined = torch.cat([text_features, meta_features, graph_encoded], dim=1)
        logits = self.fusion(combined)
        
        return logits.squeeze(-1)
    
    def predict_proba(self, 
                     input_ids: torch.Tensor,
                     attention_mask: torch.Tensor,
                     metadata: torch.Tensor,
                     graph_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get probability scores"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask, metadata, graph_features)
            probs = torch.sigmoid(logits)
        return probs
    
    def load_phase1_weights(self, checkpoint_path: str, strict: bool = False):
        """
        Load weights from Phase 1 model
        Useful for transfer learning
        
        Args:
            checkpoint_path: Path to Phase 1 checkpoint
            strict: If False, ignores missing graph encoder weights
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Filter out graph-related weights (they won't exist in Phase 1)
        model_dict = self.state_dict()
        filtered_dict = {}
        
        for k, v in state_dict.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    filtered_dict[k] = v
                else:
                    logger.warning(f"Shape mismatch for {k}: {v.shape} vs {model_dict[k].shape}")
            else:
                logger.debug(f"Skipping {k} (not in current model)")
        
        # Load weights
        missing_keys, unexpected_keys = self.load_state_dict(filtered_dict, strict=False)
        
        if missing_keys:
            logger.info(f"Missing keys (will be trained): {len(missing_keys)}")
            for key in missing_keys[:5]:  # Show first 5
                logger.info(f"  - {key}")
        
        logger.info("Phase 1 weights loaded successfully")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores (approximation)
        Returns importance scores for each feature type
        """
        importance = {}
        
        # Text importance (from first fusion layer weights)
        fusion_weights = self.fusion[0].weight.data  # [512, 864]
        text_importance = fusion_weights[:, :768].abs().mean().item()
        meta_importance = fusion_weights[:, 768:832].abs().mean().item()
        graph_importance = fusion_weights[:, 832:].abs().mean().item()
        
        importance['text'] = text_importance
        importance['metadata'] = meta_importance
        importance['graph'] = graph_importance
        
        return importance


def create_model_v2_from_phase1(phase1_checkpoint: str, 
                                num_meta_features: int = 41) -> FakeJobDetectorV2:
    """
    Factory function to create Phase 2 model initialized with Phase 1 weights
    
    Args:
        phase1_checkpoint: Path to Phase 1 model checkpoint
        num_meta_features: Number of metadata features
        
    Returns:
        FakeJobDetectorV2 with Phase 1 weights loaded
    """
    model = FakeJobDetectorV2(num_meta_features=num_meta_features)
    model.load_phase1_weights(phase1_checkpoint)
    return model
