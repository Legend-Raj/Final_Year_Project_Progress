"""
Shared metadata feature extraction utility.
Used by hybrid_predictor_complete.py and final_fusion_model.py
to avoid code duplication.
"""
import numpy as np
from typing import List


def extract_metadata_from_job(job_dict: dict, feature_names: List[str]) -> list:
    """
    Extract metadata feature vector from a job posting dictionary,
    matching the feature names from the trained scaler.
    
    Args:
        job_dict: Job posting data with keys like 'title', 'description', etc.
        feature_names: List of feature names from the trained DataProcessor.
        
    Returns:
        List of feature values aligned with feature_names.
    """
    features = {}
    
    # Text length features
    features['desc_length'] = len(str(job_dict.get('description', '')))
    features['title_length'] = len(str(job_dict.get('title', '')))
    features['req_length'] = len(str(job_dict.get('requirements', '')))
    features['company_profile_length'] = len(str(job_dict.get('company_profile', '')))
    
    # Binary presence features
    features['has_salary'] = 1 if job_dict.get('salary_range') else 0
    features['has_department'] = 1 if job_dict.get('department') else 0
    features['has_company_logo'] = int(job_dict.get('has_company_logo', 0))
    features['has_questions'] = int(job_dict.get('has_questions', 0))
    features['telecommuting'] = int(job_dict.get('telecommuting', 0))
    
    # Location features
    features['has_location'] = 1 if job_dict.get('location') else 0
    features['is_us_location'] = 1 if 'US' in str(job_dict.get('location', '')) else 0
    
    # Employment type (one-hot)
    emp_type = job_dict.get('employment_type', 'nan')
    for fname in feature_names:
        if fname.startswith('emp_type_'):
            features[fname] = 1 if fname == f'emp_type_{emp_type}' else 0
    
    # Experience level
    exp_mapping = {
        'Not Applicable': 0, 'Entry level': 1, 'Associate': 2,
        'Mid-Senior level': 3, 'Director': 4, 'Executive': 5
    }
    features['experience_level'] = exp_mapping.get(
        job_dict.get('required_experience'), 0
    )
    
    # Education level
    edu_mapping = {
        'Unspecified': 0, 'High School or equivalent': 1, 'Vocational': 2,
        'Some College Coursework Completed': 3, 'Associate Degree': 4,
        "Bachelor's Degree": 5, "Master's Degree": 6,
        'Doctorate': 7, 'Professional': 8
    }
    features['education_level'] = edu_mapping.get(
        job_dict.get('required_education'), 0
    )
    
    # Industry (one-hot)
    industry = job_dict.get('industry', '')
    for fname in feature_names:
        if fname.startswith('industry_'):
            features[fname] = 1 if industry == fname.replace('industry_', '') else 0
    
    # Function (one-hot)
    function = job_dict.get('function', '')
    for fname in feature_names:
        if fname.startswith('function_'):
            features[fname] = 1 if function == fname.replace('function_', '') else 0
    
    # Missing value indicators
    features['desc_missing'] = 1 if not job_dict.get('description') else 0
    features['requirements_missing'] = 1 if not job_dict.get('requirements') else 0
    
    # Build vector aligned with feature_names
    return [features.get(f, 0) for f in feature_names]
