"""
PyTorch Dataset for fake job detection
"""
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer


class FakeJobDataset(Dataset):
    def __init__(self, texts, metadata, labels, tokenizer_name='distilbert-base-uncased', max_length=512):
        self.texts = texts
        self.metadata = torch.tensor(metadata, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        meta = self.metadata[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'metadata': meta,
            'label': label
        }
