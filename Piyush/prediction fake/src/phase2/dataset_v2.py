"""
PyTorch Dataset for Phase 2 (with Graph Features)
"""
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer


class FakeJobDatasetV2(Dataset):
    """
    Dataset for Phase 2 model
    Includes: text, metadata, and graph features
    """
    
    def __init__(self, 
                 texts: list,
                 metadata: list,
                 graph_features: list,
                 labels: list,
                 tokenizer_name: str = 'distilbert-base-uncased',
                 max_length: int = 512):
        """
        Args:
            texts: List of text strings
            metadata: List of metadata feature vectors
            graph_features: List of graph feature vectors (NEW)
            labels: List of labels
            tokenizer_name: HuggingFace tokenizer name
            max_length: Max sequence length for tokenizer
        """
        self.texts = texts
        self.metadata = torch.tensor(metadata, dtype=torch.float32)
        self.graph_features = torch.tensor(graph_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]
        label = self.labels[idx]
        meta = self.metadata[idx]
        graph = self.graph_features[idx]
        
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
            'graph_features': graph,  # NEW
            'label': label
        }


# Backward compatibility alias
FakeJobDatasetPhase2 = FakeJobDatasetV2
