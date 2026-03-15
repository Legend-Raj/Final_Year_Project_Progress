"""
Data Enrichment Pipeline for Phase 2
Adds graph/network features to existing Phase 1 processed data
"""
import os
import pickle
import logging
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .network_features import NetworkFeatureExtractor
from .config import get_config

logger = logging.getLogger(__name__)


class DataEnricher:
    """
    Enriches processed data with network features
    
    Usage:
        enricher = DataEnricher()
        enricher.enrich_dataset('processed/train.pkl', 'processed/train_v2.pkl')
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.config = get_config()
        self.extractor = NetworkFeatureExtractor(cache_dir=cache_dir)
        logger.info("DataEnricher initialized")
    
    def _load_phase1_data(self, path: str) -> dict:
        """Load Phase 1 processed data"""
        logger.info(f"Loading Phase 1 data from {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def _extract_graph_features(self, texts: list, df_original: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Extract graph features for all samples
        
        Args:
            texts: List of combined texts (from Phase 1)
            df_original: Original dataframe (if available) for additional columns
            
        Returns:
            Array of shape (n_samples, n_graph_features)
        """
        # We need to parse company info from texts
        # Texts are in format: "Title: X. Description: Y. Requirements: Z. Company: W. Benefits: V"
        
        features_list = []
        
        logger.info(f"Extracting graph features for {len(texts)} samples...")
        
        for idx, text in enumerate(tqdm(texts, desc="Enriching")):
            # Parse company section
            company_profile = ""
            contact_email = None
            company_name = ""
            
            # Extract company profile from text
            if "Company:" in text:
                company_start = text.find("Company:") + 8
                company_end = text.find(". Benefits:")
                if company_end == -1:
                    company_end = len(text)
                company_profile = text[company_start:company_end].strip()
            
            # Extract title as company name fallback
            if "Title:" in text:
                title_start = text.find("Title:") + 6
                title_end = text.find(". Description:")
                if title_end != -1:
                    company_name = text[title_start:title_end].strip()
            
            # Try to extract email
            import re
            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
            if email_match:
                contact_email = email_match.group()
            
            # Extract features
            features = self.extractor.extract(
                company_profile=company_profile,
                contact_email=contact_email,
                company_name=company_name
            )
            
            features_list.append(features.to_vector())
        
        return np.array(features_list, dtype=np.float32)
    
    def enrich_dataset(self, 
                      input_path: str, 
                      output_path: str,
                      df_original: Optional[pd.DataFrame] = None) -> dict:
        """
        Enrich a Phase 1 dataset with graph features
        
        Args:
            input_path: Path to Phase 1 .pkl file
            output_path: Path to save enriched .pkl file
            df_original: Optional original dataframe
            
        Returns:
            Enriched data dictionary
        """
        # Load Phase 1 data
        data = self._load_phase1_data(input_path)
        
        logger.info(f"Loaded {len(data['labels'])} samples")
        
        # Extract graph features
        graph_features = self._extract_graph_features(data['texts'], df_original)
        
        logger.info(f"Extracted graph features shape: {graph_features.shape}")
        
        # Create enriched data
        enriched_data = {
            'indices': data['indices'],
            'texts': data['texts'],
            'metadata': data['metadata'],
            'graph_features': graph_features,  # NEW
            'labels': data['labels']
        }
        
        # Save enriched data
        logger.info(f"Saving enriched data to {output_path}")
        with open(output_path, 'wb') as f:
            pickle.dump(enriched_data, f)
        
        logger.info("Enrichment complete!")
        
        # Print statistics
        self._print_feature_stats(graph_features)
        
        return enriched_data
    
    def _print_feature_stats(self, graph_features: np.ndarray):
        """Print statistics about graph features"""
        from .network_features import NetworkFeatures
        
        feature_names = NetworkFeatures.get_feature_names()
        
        print("\n" + "="*60)
        print("GRAPH FEATURE STATISTICS")
        print("="*60)
        
        for i, name in enumerate(feature_names):
            values = graph_features[:, i]
            print(f"{name:30s}: mean={values.mean():.3f}, std={values.std():.3f}, "
                  f"min={values.min():.3f}, max={values.max():.3f}")
        
        print("="*60)
    
    def enrich_all_splits(self, processed_dir: str = 'processed'):
        """
        Enrich all dataset splits (train, val, test)
        
        Args:
            processed_dir: Directory containing Phase 1 processed files
        """
        splits = ['train', 'val', 'test']
        
        for split in splits:
            input_path = os.path.join(processed_dir, f'{split}.pkl')
            output_path = os.path.join(processed_dir, f'{split}_v2.pkl')
            
            if os.path.exists(input_path):
                print(f"\n{'='*60}")
                print(f"Processing {split.upper()}")
                print(f"{'='*60}")
                self.enrich_dataset(input_path, output_path)
            else:
                logger.warning(f"Input file not found: {input_path}")
        
        print("\n" + "="*60)
        print("All datasets enriched successfully!")
        print("Phase 2 files created:")
        for split in splits:
            print(f"  - processed/{split}_v2.pkl")
        print("="*60)


def main():
    """CLI for data enrichment"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enrich Phase 1 data with graph features')
    parser.add_argument('--processed-dir', default='processed', 
                       help='Directory containing processed data')
    parser.add_argument('--split', default='all', 
                       choices=['train', 'val', 'test', 'all'],
                       help='Which split to enrich')
    
    args = parser.parse_args()
    
    enricher = DataEnricher()
    
    if args.split == 'all':
        enricher.enrich_all_splits(args.processed_dir)
    else:
        input_path = os.path.join(args.processed_dir, f'{args.split}.pkl')
        output_path = os.path.join(args.processed_dir, f'{args.split}_v2.pkl')
        enricher.enrich_dataset(input_path, output_path)


if __name__ == '__main__':
    main()
