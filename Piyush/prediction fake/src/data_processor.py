"""
Data preprocessing for fake job detection
Handles: Cleaning, feature extraction, train/val/test split
"""
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os


class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_data(self, filepath):
        """Load and basic cleaning"""
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records")
        print(f"Fraud distribution:\n{df['fraudulent'].value_counts()}")
        return df
    
    def clean_text(self, text):
        """Basic text cleaning"""
        if pd.isna(text):
            return ""
        text = str(text)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        return text.strip().lower()
    
    def extract_metadata_features(self, df):
        """
        Extract numerical metadata features
        """
        features = pd.DataFrame()
        
        # 1. Text length features
        features['desc_length'] = df['description'].fillna('').apply(len)
        features['title_length'] = df['title'].fillna('').apply(len)
        features['req_length'] = df['requirements'].fillna('').apply(len)
        features['company_profile_length'] = df['company_profile'].fillna('').apply(len)
        
        # 2. Binary presence features
        features['has_salary'] = df['salary_range'].notna().astype(int)
        features['has_department'] = df['department'].notna().astype(int)
        features['has_company_logo'] = df['has_company_logo'].fillna(0).astype(int)
        features['has_questions'] = df['has_questions'].fillna(0).astype(int)
        features['telecommuting'] = df['telecommuting'].fillna(0).astype(int)
        
        # 3. Location features
        features['has_location'] = df['location'].notna().astype(int)
        features['is_us_location'] = df['location'].fillna('').str.contains('US', case=False).astype(int)
        
        # 4. Employment type encoding (simplified)
        emp_type_dummies = pd.get_dummies(df['employment_type'], prefix='emp_type', dummy_na=True)
        features = pd.concat([features, emp_type_dummies], axis=1)
        
        # 5. Required experience encoding
        exp_mapping = {
            'Not Applicable': 0,
            'Entry level': 1,
            'Associate': 2,
            'Mid-Senior level': 3,
            'Director': 4,
            'Executive': 5
        }
        features['experience_level'] = df['required_experience'].map(exp_mapping).fillna(0)
        
        # 6. Required education encoding
        edu_mapping = {
            'Unspecified': 0,
            'High School or equivalent': 1,
            'Vocational': 2,
            'Some College Coursework Completed': 3,
            'Associate Degree': 4,
            "Bachelor's Degree": 5,
            "Master's Degree": 6,
            'Doctorate': 7,
            'Professional': 8
        }
        features['education_level'] = df['required_education'].map(edu_mapping).fillna(0)
        
        # 7. Industry and Function (top categories only)
        top_industries = df['industry'].value_counts().head(10).index
        for ind in top_industries:
            features[f'industry_{ind}'] = (df['industry'] == ind).astype(int)
        
        top_functions = df['function'].value_counts().head(10).index
        for func in top_functions:
            features[f'function_{func}'] = (df['function'] == func).astype(int)
        
        # 8. Missing value indicators
        features['desc_missing'] = df['description'].isna().astype(int)
        features['requirements_missing'] = df['requirements'].isna().astype(int)
        
        self.feature_names = features.columns.tolist()
        return features
    
    def prepare_text(self, df):
        """Combine text fields"""
        text_data = []
        for _, row in df.iterrows():
            combined = f"Title: {row['title']}. "
            combined += f"Description: {self.clean_text(row['description'])}. "
            combined += f"Requirements: {self.clean_text(row['requirements'])}. "
            combined += f"Company: {self.clean_text(row['company_profile'])}. "
            combined += f"Benefits: {self.clean_text(row['benefits'])}"
            text_data.append(combined)
        return text_data
    
    def preprocess(self, filepath, save_dir='processed'):
        """Full preprocessing pipeline"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Load
        df = self.load_data(filepath)
        
        # Extract features
        print("Extracting metadata features...")
        meta_features = self.extract_metadata_features(df)
        
        print("Preparing text...")
        text_data = self.prepare_text(df)
        
        # Labels
        labels = df['fraudulent'].values
        
        # Scale metadata
        meta_scaled = self.scaler.fit_transform(meta_features)
        
        # Temporal split (important for fraud detection)
        # Sort by job_id (assuming it's roughly chronological)
        df_sorted = df.sort_values('job_id')
        sorted_indices = df_sorted.index.values
        
        # Split: 70% train, 15% val, 15% test
        n = len(sorted_indices)
        train_idx = sorted_indices[:int(0.7 * n)]
        val_idx = sorted_indices[int(0.7 * n):int(0.85 * n)]
        test_idx = sorted_indices[int(0.85 * n):]
        
        # Create datasets
        splits = {
            'train': (train_idx, [text_data[i] for i in train_idx], meta_scaled[train_idx], labels[train_idx]),
            'val': (val_idx, [text_data[i] for i in val_idx], meta_scaled[val_idx], labels[val_idx]),
            'test': (test_idx, [text_data[i] for i in test_idx], meta_scaled[test_idx], labels[test_idx])
        }
        
        # Save
        for name, (indices, texts, meta, y) in splits.items():
            data = {
                'indices': indices,
                'texts': texts,
                'metadata': meta,
                'labels': y
            }
            with open(f'{save_dir}/{name}.pkl', 'wb') as f:
                pickle.dump(data, f)
            print(f"{name}: {len(y)} samples, {sum(y)} frauds ({100*sum(y)/len(y):.1f}%)")
        
        # Save scaler and feature names
        with open(f'{save_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(f'{save_dir}/feature_names.pkl', 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        print(f"\nPreprocessing complete! Data saved to {save_dir}/")
        return splits


if __name__ == "__main__":
    processor = DataProcessor()
    processor.preprocess('fake_job_postings.csv')
