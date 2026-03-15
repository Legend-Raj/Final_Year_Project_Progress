"""
Phase 2 Training Script
Trains model with text + metadata + graph features
"""
import os
import sys
import pickle
import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2.model_v2 import FakeJobDetectorV2, create_model_v2_from_phase1
from phase2.dataset_v2 import FakeJobDatasetV2
from phase2.config import get_config, Phase2Config
from focal_loss import FocalLoss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainerPhase2:
    """Trainer for Phase 2 model"""
    
    def __init__(self, model: FakeJobDetectorV2, config: Phase2Config, 
                 device: torch.device, loss_type: str = 'bce'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Setup optimizer with different learning rates
        bert_params = list(model.distilbert.named_parameters())
        meta_params = list(model.metadata_encoder.named_parameters())
        graph_params = list(model.graph_encoder.named_parameters())
        fusion_params = list(model.fusion.named_parameters())
        
        self.optimizer = optim.AdamW([
            {
                'params': [p for n, p in bert_params],
                'lr': config.learning_rate_bert,
                'weight_decay': 0.01
            },
            {
                'params': meta_params + graph_params + fusion_params,
                'lr': config.learning_rate_other,
                'weight_decay': 0.01
            }
        ])
        
        # Loss function
        if loss_type == 'focal':
            self.criterion = FocalLoss(alpha=0.75, gamma=2.0)
            logger.info("Using Focal Loss (alpha=0.75, gamma=2.0)")
        else:
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([10.0]).to(device)
            )
            logger.info("Using BCEWithLogitsLoss (pos_weight=10.0)")
        
        self.best_val_auc = 0.0
        self.patience_counter = 0
        
        logger.info(f"Trainer initialized on {device}")
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            metadata = batch['metadata'].to(self.device)
            graph_features = batch['graph_features'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            logits = self.model(
                input_ids, 
                attention_mask, 
                metadata,
                graph_features
            )
            loss = self.criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        auc = roc_auc_score(all_labels, all_preds)
        
        return avg_loss, auc
    
    def evaluate(self, dataloader: DataLoader) -> Dict:
        """Evaluate on validation/test set"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                metadata = batch['metadata'].to(self.device)
                graph_features = batch['graph_features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(
                    input_ids,
                    attention_mask,
                    metadata,
                    graph_features
                )
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                all_preds.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        auc = roc_auc_score(all_labels, all_preds)
        
        # Find best F1 threshold
        best_f1 = 0.0
        best_thresh = 0.5
        for thresh in np.arange(0.1, 0.9, 0.05):
            preds_binary = (np.array(all_preds) >= thresh).astype(int)
            f1 = f1_score(all_labels, preds_binary, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        # Metrics at best threshold
        preds_binary = (np.array(all_preds) >= best_thresh).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, preds_binary, average='binary', zero_division=0
        )
        
        return {
            'loss': avg_loss,
            'auc': auc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'threshold': best_thresh
        }
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_val_auc': self.best_val_auc,
            'config': self.config.to_dict()
        }, path)
        logger.info(f"Checkpoint saved to {path}")


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Phase 2 model')
    parser.add_argument('--processed-dir', default='processed',
                       help='Directory with processed data')
    parser.add_argument('--use-phase1', default='models/best_model_full.pt',
                       help='Path to Phase 1 checkpoint for transfer learning')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--lr-bert', type=float, default=2e-5)
    parser.add_argument('--lr-other', type=float, default=2e-4)
    parser.add_argument('--no-transfer', action='store_true',
                       help='Train from scratch without Phase 1 weights')
    parser.add_argument('--loss', choices=['bce', 'focal'], default='bce',
                       help='Loss function: bce (default) or focal')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load config
    config = Phase2Config(
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate_bert=args.lr_bert,
        learning_rate_other=args.lr_other
    )
    
    # Load enriched data
    logger.info("Loading Phase 2 data...")
    train_data = pickle.load(open(f'{args.processed_dir}/train_v2.pkl', 'rb'))
    val_data = pickle.load(open(f'{args.processed_dir}/val_v2.pkl', 'rb'))
    
    # Create datasets
    train_dataset = FakeJobDatasetV2(
        train_data['texts'],
        train_data['metadata'],
        train_data['graph_features'],
        train_data['labels']
    )
    val_dataset = FakeJobDatasetV2(
        val_data['texts'],
        val_data['metadata'],
        val_data['graph_features'],
        val_data['labels']
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size
    )
    
    # Get dimensions
    num_meta_features = train_data['metadata'].shape[1]
    num_graph_features = train_data['graph_features'].shape[1]
    
    logger.info(f"Metadata features: {num_meta_features}")
    logger.info(f"Graph features: {num_graph_features}")
    
    # Create model
    if args.no_transfer or not os.path.exists(args.use_phase1):
        logger.info("Training from scratch...")
        model = FakeJobDetectorV2(
            num_meta_features=num_meta_features,
            num_graph_features=num_graph_features
        )
    else:
        logger.info(f"Loading Phase 1 weights from {args.use_phase1}")
        model = create_model_v2_from_phase1(
            args.use_phase1,
            num_meta_features=num_meta_features
        )
    
    # Show model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = TrainerPhase2(model, config, device, loss_type=args.loss)
    
    # Training loop
    os.makedirs('models', exist_ok=True)
    
    logger.info(f"\n{'='*60}")
    logger.info("STARTING PHASE 2 TRAINING")
    logger.info(f"{'='*60}")
    
    for epoch in range(config.epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{config.epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        train_loss, train_auc = trainer.train_epoch(train_loader)
        logger.info(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
        
        # Validate
        val_metrics = trainer.evaluate(val_loader)
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"Val AUC: {val_metrics['auc']:.4f}")
        logger.info(f"Val F1: {val_metrics['f1']:.4f} (thresh={val_metrics['threshold']:.2f})")
        logger.info(f"Val Precision: {val_metrics['precision']:.4f}")
        logger.info(f"Val Recall: {val_metrics['recall']:.4f}")
        
        # Feature importance
        importance = model.get_feature_importance()
        logger.info(f"Feature importance - Text: {importance['text']:.4f}, "
                   f"Meta: {importance['metadata']:.4f}, "
                   f"Graph: {importance['graph']:.4f}")
        
        # Save best
        if val_metrics['auc'] > trainer.best_val_auc:
            trainer.best_val_auc = val_metrics['auc']
            trainer.patience_counter = 0
            trainer.save_checkpoint('models/best_model_phase2.pt', epoch, val_metrics)
            logger.info(f"✓ New best model! AUC: {val_metrics['auc']:.4f}")
        else:
            trainer.patience_counter += 1
            logger.info(f"Patience: {trainer.patience_counter}/{config.patience}")
            
            if trainer.patience_counter >= config.patience:
                logger.info("Early stopping triggered!")
                break
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training complete! Best Val AUC: {trainer.best_val_auc:.4f}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
