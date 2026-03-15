"""
DistilBERT + Metadata Fusion Model (Phase 1)
"""
import torch
import torch.nn as nn
from transformers import DistilBertModel


class FakeJobDetector(nn.Module):
    def __init__(self, num_meta_features, dropout_rate=0.4):
        super(FakeJobDetector, self).__init__()
        
        # 1. DistilBERT for text encoding
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.text_dropout = nn.Dropout(0.2)
        
        # Freeze bottom 4 layers (optional - saves memory and prevents overfitting)
        for param in self.distilbert.transformer.layer[:4].parameters():
            param.requires_grad = False
        
        # DistilBERT output dim = 768
        
        # 2. Metadata encoder (YOUR custom layer)
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
        
        # 3. Fusion layers (YOUR custom brain!)
        combined_dim = 768 + 64  # text + metadata
        
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
            
            nn.Linear(128, 1)  # Raw logits (sigmoid in loss)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize custom layers"""
        for module in [self.metadata_encoder, self.fusion]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, input_ids, attention_mask, metadata):
        """
        Forward pass
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            metadata: [batch, num_meta_features]
        """
        # Text encoding via DistilBERT
        bert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use [CLS] token representation
        text_features = bert_output.last_hidden_state[:, 0, :]  # [batch, 768]
        text_features = self.text_dropout(text_features)
        
        # Metadata encoding
        meta_features = self.metadata_encoder(metadata)  # [batch, 64]
        
        # Fusion - concatenate both
        combined = torch.cat([text_features, meta_features], dim=1)  # [batch, 832]
        
        # Final prediction
        logits = self.fusion(combined)  # [batch, 1]
        
        return logits.squeeze(-1)  # [batch]
    
    def predict_proba(self, input_ids, attention_mask, metadata):
        """Get probability scores"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask, metadata)
            probs = torch.sigmoid(logits)
        return probs
