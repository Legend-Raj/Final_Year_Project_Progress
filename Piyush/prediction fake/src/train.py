"""
Training script for Phase 1
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support

from dataset import FakeJobDataset
from model import FakeJobDetector


class Trainer:
    def __init__(self, model, device, learning_rate=2e-5, weight_decay=0.01):
        self.model = model.to(device)
        self.device = device
        
        # Separate learning rates for BERT and custom layers
        bert_params = list(model.distilbert.named_parameters())
        other_params = [
            ('metadata_encoder', p) for n, p in model.metadata_encoder.named_parameters()
        ] + [
            ('fusion', p) for n, p in model.fusion.named_parameters()
        ]
        
        # Optimizer with different learning rates
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in bert_params],
                'lr': learning_rate,  # Lower LR for pre-trained
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in other_params],
                'lr': learning_rate * 10,  # Higher LR for new layers
                'weight_decay': weight_decay
            }
        ]
        
        self.optimizer = optim.AdamW(optimizer_grouped_parameters)
        
        # Loss with class weights (handle imbalance)
        # Fake = minority class, weight it higher
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))
        
        self.best_val_auc = 0
        self.patience_counter = 0
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            metadata = batch['metadata'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            logits = self.model(input_ids, attention_mask, metadata)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        auc = roc_auc_score(all_labels, all_preds)
        
        return avg_loss, auc
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                metadata = batch['metadata'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(input_ids, attention_mask, metadata)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        auc = roc_auc_score(all_labels, all_preds)
        
        # Find best F1 threshold
        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.arange(0.1, 0.9, 0.05):
            preds_binary = (np.array(all_preds) > thresh).astype(int)
            f1 = f1_score(all_labels, preds_binary, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        # Metrics at best threshold
        preds_binary = (np.array(all_preds) > best_thresh).astype(int)
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
    
    def save_checkpoint(self, path, epoch, metrics):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_val_auc': self.best_val_auc
        }, path)
        print(f"Checkpoint saved to {path}")


def main():
    # Configuration
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 2e-5
    PATIENCE = 3
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load processed data
    print("Loading data...")
    train_data = pickle.load(open('processed/train.pkl', 'rb'))
    val_data = pickle.load(open('processed/val.pkl', 'rb'))
    
    # Create datasets
    train_dataset = FakeJobDataset(
        train_data['texts'],
        train_data['metadata'],
        train_data['labels']
    )
    val_dataset = FakeJobDataset(
        val_data['texts'],
        val_data['metadata'],
        val_data['labels']
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    num_meta_features = train_data['metadata'].shape[1]
    print(f"Number of metadata features: {num_meta_features}")
    
    # Create model
    model = FakeJobDetector(num_meta_features=num_meta_features)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(model, device, learning_rate=LEARNING_RATE)
    
    # Training loop
    os.makedirs('models', exist_ok=True)
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"{'='*50}")
        
        # Train
        train_loss, train_auc = trainer.train_epoch(train_loader)
        print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
        
        # Validate
        val_metrics = trainer.evaluate(val_loader)
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val AUC: {val_metrics['auc']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f} (thresh={val_metrics['threshold']:.2f})")
        print(f"Val Precision: {val_metrics['precision']:.4f}")
        print(f"Val Recall: {val_metrics['recall']:.4f}")
        
        # Save best model
        if val_metrics['auc'] > trainer.best_val_auc:
            trainer.best_val_auc = val_metrics['auc']
            trainer.patience_counter = 0
            trainer.save_checkpoint('models/best_model.pt', epoch, val_metrics)
            print(f"✓ New best model! AUC: {val_metrics['auc']:.4f}")
        else:
            trainer.patience_counter += 1
            print(f"Patience: {trainer.patience_counter}/{PATIENCE}")
            
            if trainer.patience_counter >= PATIENCE:
                print("Early stopping triggered!")
                break
    
    print(f"\nTraining complete! Best Val AUC: {trainer.best_val_auc:.4f}")


if __name__ == "__main__":
    main()
