"""
Focal Loss for imbalanced binary classification.

Focal loss down-weights easy, well-classified examples and focuses
training on hard-to-classify examples (the missed frauds).

Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

Where:
  - p_t = p if y=1, else (1-p)
  - alpha balances positive/negative classes
  - gamma focuses on hard examples (gamma=0 reduces to standard BCE)

References:
  Lin et al., "Focal Loss for Dense Object Detection" (2017)

Usage:
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    loss = criterion(logits, labels)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary Focal Loss.
    
    Args:
        alpha: Weight for positive class (fraud). Higher = more focus on fraud.
               Default 0.75 means 75% weight on fraud, 25% on legitimate.
        gamma: Focusing parameter. Higher = more focus on hard examples.
               gamma=0 is standard BCE, gamma=2 is recommended default.
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, 
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output (before sigmoid), shape [batch]
            targets: Binary labels (0 or 1), shape [batch]
        
        Returns:
            Focal loss value
        """
        probs = torch.sigmoid(logits)
        
        # p_t = p if y=1, else 1-p
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # alpha_t = alpha if y=1, else 1-alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # BCE component: -log(p_t)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Focal loss = alpha_t * focal_weight * BCE
        loss = alpha_t * focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class FocalLossWithSmoothing(nn.Module):
    """
    Focal Loss with optional label smoothing.
    
    Label smoothing helps prevent overconfidence:
    - target 1.0 becomes 0.95
    - target 0.0 becomes 0.05
    """
    
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0,
                 smoothing: float = 0.05, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply label smoothing
        targets_smooth = targets * (1 - self.smoothing) + (1 - targets) * self.smoothing
        
        probs = torch.sigmoid(logits)
        p_t = probs * targets_smooth + (1 - probs) * (1 - targets_smooth)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        bce = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction='none')
        loss = alpha_t * focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
