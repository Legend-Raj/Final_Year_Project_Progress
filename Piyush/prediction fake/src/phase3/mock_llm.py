"""
Mock LLM for testing without API calls
Simulates GPT-4o-mini responses based on heuristics
"""
import re
from typing import List

from .llm_analyzer_v2 import LLMAnalysis


class MockLLMAnalyzer:
    """
    Mock LLM that simulates responses without API calls
    Useful for testing and development
    """
    
    def __init__(self):
        print("Mock LLM Analyzer initialized (NO API COSTS)")
        self.cost_per_call = 0.0
    
    def analyze(self, job_dict: dict, model_prob: float = None) -> LLMAnalysis:
        """
        Simulate LLM analysis based on heuristics
        """
        red_flags = []
        recommendations = []
        
        # Extract text
        title = str(job_dict.get('title', '')).lower()
        desc = str(job_dict.get('description', '')).lower()
        company = str(job_dict.get('company_profile', '')).lower()
        email = str(job_dict.get('contact_email', '')).lower()
        combined = title + ' ' + desc + ' ' + company
        
        # Check for red flags
        
        # 1. Money-related red flags
        money_patterns = [
            (r'\$\d{3,}/day', 'Unrealistic daily pay'),
            (r'\$\d{3,}/hour', 'Unrealistic hourly pay'),
            (r'\$\d{4,}/week', 'Unrealistic weekly pay'),
            (r'guaranteed.*\$', 'Guaranteed earnings promise'),
            (r'earn.*\$\d{3,}', 'Unrealistic earning claims'),
            (r'\$\d{2,}\+.*week', 'Too good to be true salary'),
        ]
        
        for pattern, flag in money_patterns:
            if re.search(pattern, combined):
                red_flags.append(flag)
        
        # 2. Upfront payment
        upfront_patterns = [
            (r'send.*\$\d+', 'Requests upfront payment'),
            (r'pay.*\$\d+.*(?:starter|training|kit)', 'Pay for starter kit'),
            (r'(?:registration|processing).*(?:fee|payment)', 'Registration fee required'),
            (r'buy.*equipment', 'Must buy equipment'),
            (r'investment.*required', 'Investment required'),
        ]
        
        for pattern, flag in upfront_patterns:
            if re.search(pattern, combined):
                red_flags.append(flag)
        
        # 3. Experience requirements
        if re.search(r'no experience.*\$\d{3,}', combined):
            red_flags.append('High pay for no experience')
        
        if re.search(r'no (?:skills?|qualifications?).*required', combined):
            red_flags.append('No qualifications required for high pay')
        
        # 4. Urgency/pressure
        urgency_words = ['urgent', 'immediate start', 'act now', 'limited spots', 
                        'hiring today', 'start tomorrow', 'apply now']
        for word in urgency_words:
            if word in combined:
                red_flags.append(f'Urgency pressure: "{word}"')
                break
        
        # 5. Email domain
        if '@gmail.com' in email or '@yahoo.com' in email or '@hotmail.com' in email:
            red_flags.append('Free email provider used')
        
        # 6. Vague description
        if len(desc) < 100 and re.search(r'\$\d+', combined):
            red_flags.append('Vague description with salary mention')
        
        # 7. Work from home + money
        if 'work from home' in combined and re.search(r'\$\d{3,}', combined):
            red_flags.append('WFH with unrealistic pay')
        
        # 8. Company profile issues
        if len(company) < 50:
            red_flags.append('Very short company profile')
        
        if re.search(r'\b(?:quick|easy|fast)\s+(?:money|cash|income)', combined):
            red_flags.append('Quick money promises')
        
        # Calculate probability based on red flags
        base_score = 0.1  # Start low
        
        # Each red flag adds to score
        for flag in red_flags:
            if 'upfront' in flag or 'payment' in flag:
                base_score += 0.25  # Heavy weight
            elif 'unrealistic' in flag.lower() or 'unrealistic' in flag.lower():
                base_score += 0.20
            elif 'free email' in flag:
                base_score += 0.10
            else:
                base_score += 0.15
        
        # Cap at 0.95
        fraud_probability = min(0.95, base_score)
        
        # Determine confidence
        if len(red_flags) >= 4:
            confidence = "high"
        elif len(red_flags) >= 2:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Generate reasoning
        if red_flags:
            reasoning = f"Analysis found {len(red_flags)} red flags:\n"
            for i, flag in enumerate(red_flags[:5], 1):
                reasoning += f"{i}. {flag}\n"
            
            if len(red_flags) > 5:
                reasoning += f"...and {len(red_flags) - 5} more.\n"
            
            reasoning += f"\nOverall fraud probability: {fraud_probability:.1%}"
        else:
            reasoning = "No significant red flags detected. Job posting appears legitimate."
            fraud_probability = max(0.05, fraud_probability)  # Minimum 5%
        
        # Generate recommendations
        if fraud_probability > 0.7:
            recommendations = [
                "HIGH RISK: Avoid this job posting",
                "Do not share personal information",
                "Do not make any payments",
                "Report if found on job platforms"
            ]
        elif fraud_probability > 0.4:
            recommendations = [
                "MEDIUM RISK: Proceed with caution",
                "Research company independently",
                "Verify through official channels",
                "Never pay upfront fees"
            ]
        else:
            recommendations = [
                "LOW RISK: Appears legitimate",
                "Still verify company before applying",
                "Use standard job search precautions"
            ]
        
        return LLMAnalysis(
            fraud_probability=fraud_probability,
            confidence=confidence,
            reasoning=reasoning,
            red_flags=red_flags,
            recommendations=recommendations,
            cost_usd=0.0,  # Free!
            cached=False,
            provider="mock"
        )


# For testing
if __name__ == "__main__":
    analyzer = MockLLMAnalyzer()
    
    # Test suspicious job
    suspicious = {
        'title': 'Work From Home - $500/Day Guaranteed!',
        'description': 'No experience needed! Send $50 for starter kit. Earn easy money!',
        'requirements': 'No requirements',
        'company_profile': 'Quick money opportunities!',
        'contact_email': 'quickmoney@gmail.com'
    }
    
    result = analyzer.analyze(suspicious)
    print(f"Fraud: {result.fraud_probability:.1%}")
    print(f"Confidence: {result.confidence}")
    print(f"Red flags: {result.red_flags}")
    print(f"\n{result.reasoning}")
