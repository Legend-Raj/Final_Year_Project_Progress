"""
Network/Graph feature extraction for job postings
Extracts company reputation and network-based signals
"""
import re
import time
import hashlib
import pickle
import os
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from urllib.parse import urlparse
from datetime import datetime, timezone
import logging

import requests
import numpy as np
import pandas as pd

from .config import get_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NetworkFeatures:
    """Container for network-based features"""
    # Domain features
    domain_age_days: float = 0.0
    domain_has_ssl: int = 0
    domain_suspicious_tld: int = 0
    
    # Email features
    email_is_corporate: int = 0
    email_is_free_provider: int = 0
    email_domain_age_days: float = 0.0
    
    # Company presence
    has_linkedin_page: int = 0
    linkedin_employee_count: float = 0.0
    linkedin_followers: float = 0.0
    
    # Website quality
    website_has_contact_page: int = 0
    website_has_career_page: int = 0
    website_professional_score: float = 0.0
    
    # Reputation
    domain_trust_score: float = 0.5  # Default neutral
    is_known_fake_domain: int = 0
    
    # WHOIS enriched (from real API)
    whois_domain_age_days: float = -1.0   # -1 means not looked up
    whois_registrar: str = ""             # Registrar name from WHOIS
    whois_expires_soon: int = 0           # 1 if domain expires within 90 days
    
    def to_vector(self) -> np.ndarray:
        """Convert to numpy vector for model input"""
        return np.array([
            self.domain_age_days / 365.0,  # Normalize to years
            self.domain_has_ssl,
            self.domain_suspicious_tld,
            self.email_is_corporate,
            self.email_is_free_provider,
            self.email_domain_age_days / 365.0,
            self.has_linkedin_page,
            np.log1p(self.linkedin_employee_count),
            np.log1p(self.linkedin_followers),
            self.website_has_contact_page,
            self.website_has_career_page,
            self.website_professional_score,
            self.domain_trust_score,
            self.is_known_fake_domain,
        ], dtype=np.float32)
    
    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Get list of feature names"""
        return [
            'domain_age_years',
            'domain_has_ssl',
            'domain_suspicious_tld',
            'email_is_corporate',
            'email_is_free_provider',
            'email_domain_age_years',
            'has_linkedin_page',
            'linkedin_employee_count_log',
            'linkedin_followers_log',
            'website_has_contact_page',
            'website_has_career_page',
            'website_professional_score',
            'domain_trust_score',
            'is_known_fake_domain',
        ]


class NetworkFeatureExtractor:
    """
    Extracts network-based features for fake job detection
    
    Features extracted:
    - Domain age and reputation
    - Email domain analysis
    - Company LinkedIn presence
    - Website quality heuristics
    """
    
    # Known suspicious patterns
    SUSPICIOUS_TLDS = ['.tk', '.ml', '.ga', '.cf', '.top', '.xyz', '.click', '.link']
    FREE_EMAIL_PROVIDERS = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 
                           'aol.com', 'icloud.com', 'protonmail.com', 'mail.com']
    KNOWN_FAKE_PATTERNS = [
        'quick-apply', 'easy-money', 'instant-hire', 'work-from-home-jobs',
        'no-experience-required', 'get-rich', 'earn-today'
    ]
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.config = get_config()
        self.cache_dir = cache_dir or self.config.cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # WHOIS API key
        self.whois_api_key = self.config.whois_api_key
        
        # In-memory cache
        self._cache: Dict[str, Tuple[NetworkFeatures, float]] = {}
        self._whois_cache: Dict[str, dict] = {}  # domain -> WHOIS response
        
        # Load persistent cache if exists
        self._persistent_cache_path = os.path.join(self.cache_dir, 'network_features_cache.pkl')
        self._whois_cache_path = os.path.join(self.cache_dir, 'whois_cache.pkl')
        self._load_cache()
        self._load_whois_cache()
        
        whois_status = "enabled" if self.whois_api_key else "disabled (no WHOIS_API_KEY)"
        logger.info(f"NetworkFeatureExtractor initialized (cache: {self.cache_dir}, WHOIS: {whois_status})")
        print(f"\n{'='*60}")
        print(f"[WHOIS] API Key: {'✓ LOADED (' + self.whois_api_key[:8] + '...)' if self.whois_api_key else '✗ NOT SET'}")
        print(f"{'='*60}\n")
    
    def _get_cache_key(self, company_profile: str, contact_email: Optional[str]) -> str:
        """Generate cache key from company info"""
        content = f"{company_profile}_{contact_email}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _load_cache(self):
        """Load persistent cache from disk"""
        if os.path.exists(self._persistent_cache_path):
            try:
                with open(self._persistent_cache_path, 'rb') as f:
                    self._cache = pickle.load(f)
                logger.info(f"Loaded {len(self._cache)} cached entries")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self._cache = {}
    
    def _load_whois_cache(self):
        """Load WHOIS response cache from disk"""
        if os.path.exists(self._whois_cache_path):
            try:
                with open(self._whois_cache_path, 'rb') as f:
                    self._whois_cache = pickle.load(f)
                logger.info(f"Loaded {len(self._whois_cache)} WHOIS cache entries")
            except Exception as e:
                logger.warning(f"Failed to load WHOIS cache: {e}")
                self._whois_cache = {}
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self._persistent_cache_path, 'wb') as f:
                pickle.dump(self._cache, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _save_whois_cache(self):
        """Save WHOIS cache to disk"""
        try:
            with open(self._whois_cache_path, 'wb') as f:
                pickle.dump(self._whois_cache, f)
        except Exception as e:
            logger.warning(f"Failed to save WHOIS cache: {e}")
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cache entry is still valid"""
        ttl_seconds = self.config.domain_cache_ttl * 3600
        return (time.time() - timestamp) < ttl_seconds
    
    def _extract_domain_from_text(self, text: str) -> Optional[str]:
        """Extract domain from company profile or description"""
        if pd.isna(text):
            return None
        
        # Look for URLs
        url_pattern = r'https?://(?:www\.)?([^/\s]+)'
        matches = re.findall(url_pattern, str(text))
        
        if matches:
            return matches[0].lower()
        
        # Look for domain-like patterns
        domain_pattern = r'\b([a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z]{2,})\b'
        matches = re.findall(domain_pattern, str(text))
        
        if matches:
            return matches[0].lower()
        
        return None
    
    def _extract_email_domain(self, email: Optional[str]) -> Optional[str]:
        """Extract domain from email address"""
        if pd.isna(email) or not email:
            return None
        
        email = str(email).lower().strip()
        if '@' in email:
            return email.split('@')[1]
        
        return None
    
    def _check_ssl(self, domain: str) -> int:
        """Check if domain likely has SSL (heuristic)"""
        # In production, this would make an HTTP request
        # For now, assume legitimate domains have SSL
        return 1 if not any(tld in domain for tld in ['.tk', '.ml', '.ga']) else 0
    
    def _check_suspicious_tld(self, domain: str) -> int:
        """Check if domain uses suspicious TLD"""
        return int(any(domain.endswith(tld) for tld in self.SUSPICIOUS_TLDS))
    
    def _whois_lookup(self, domain: str) -> Optional[dict]:
        """
        Look up domain via WHOIS XML API.
        Returns parsed dict with: created_date, expires_date, registrar, age_days, expires_soon
        Falls back to None on any failure.
        """
        if not self.whois_api_key:
            print(f"\n[WHOIS] ⚠ No API key set — skipping lookup for: {domain}")
            return None
        
        # Check WHOIS cache first
        if domain in self._whois_cache:
            cached = self._whois_cache[domain]
            cache_age = time.time() - cached.get('_cached_at', 0)
            if cache_age < self.config.domain_cache_ttl * 3600:
                print(f"\n[WHOIS] ✓ Cache hit for: {domain} (cached {cache_age/3600:.1f}h ago)")
                print(f"[WHOIS]   Age: {cached.get('age_days', 'N/A')} days | Registrar: {cached.get('registrar', 'N/A')}")
                return cached
        
        try:
            url = "https://www.whoisxmlapi.com/whoisserver/WhoisService"
            params = {
                'apiKey': self.whois_api_key,
                'domainName': domain,
                'outputFormat': 'JSON',
            }
            
            print(f"\n{'='*60}")
            print(f"[WHOIS] 🔍 API CALL for domain: {domain}")
            print(f"[WHOIS] URL: {url}?domainName={domain}&outputFormat=JSON")
            print(f"{'='*60}")
            
            resp = requests.get(url, params=params, timeout=self.config.request_timeout)
            
            print(f"[WHOIS] HTTP Status: {resp.status_code}")
            
            data = resp.json()
            
            # Check for API errors in response
            if 'ErrorMessage' in data:
                err = data['ErrorMessage']
                print(f"[WHOIS] ❌ API ERROR: {err.get('msg', 'Unknown error')}")
                print(f"[WHOIS]   Error Code: {err.get('errorCode', 'N/A')}")
                return None
            
            resp.raise_for_status()
            
            whois_record = data.get('WhoisRecord', {})
            
            # Print raw key fields from response
            print(f"[WHOIS] ✓ Response received!")
            print(f"[WHOIS]   Domain Name : {whois_record.get('domainName', 'N/A')}")
            print(f"[WHOIS]   Created Date: {whois_record.get('createdDate') or whois_record.get('registryData', {}).get('createdDate', 'N/A')}")
            print(f"[WHOIS]   Expires Date: {whois_record.get('expiresDate') or whois_record.get('registryData', {}).get('expiresDate', 'N/A')}")
            print(f"[WHOIS]   Updated Date: {whois_record.get('updatedDate') or whois_record.get('registryData', {}).get('updatedDate', 'N/A')}")
            print(f"[WHOIS]   Registrar   : {whois_record.get('registrarName') or whois_record.get('registryData', {}).get('registrarName', 'N/A')}")
            
            # Parse dates
            created_str = whois_record.get('createdDate') or whois_record.get('registryData', {}).get('createdDate')
            expires_str = whois_record.get('expiresDate') or whois_record.get('registryData', {}).get('expiresDate')
            registrar = whois_record.get('registrarName') or whois_record.get('registryData', {}).get('registrarName') or ''
            
            result = {
                'registrar': registrar,
                'age_days': -1.0,
                'expires_soon': 0,
                'created_date': created_str,
                'expires_date': expires_str,
                '_cached_at': time.time(),
            }
            
            now = datetime.now(timezone.utc)
            
            if created_str:
                try:
                    created_dt = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                    result['age_days'] = max(0, (now - created_dt).days)
                except (ValueError, TypeError):
                    pass
            
            if expires_str:
                try:
                    expires_dt = datetime.fromisoformat(expires_str.replace('Z', '+00:00'))
                    days_until_expiry = (expires_dt - now).days
                    result['expires_soon'] = 1 if days_until_expiry < 90 else 0
                except (ValueError, TypeError):
                    pass
            
            # Print final parsed result
            age = result['age_days']
            years = int(age // 365) if age >= 0 else 0
            months = int((age % 365) // 30) if age >= 0 else 0
            print(f"[WHOIS] 📊 PARSED RESULT:")
            print(f"[WHOIS]   Domain Age  : {age:.0f} days ({years}y {months}m)")
            print(f"[WHOIS]   Registrar   : {registrar}")
            print(f"[WHOIS]   Expires Soon: {'YES ⚠' if result['expires_soon'] else 'No'}")
            print(f"{'='*60}\n")
            
            # Cache the result
            self._whois_cache[domain] = result
            self._save_whois_cache()
            
            logger.info(f"WHOIS lookup: {domain} → age={result['age_days']}d, registrar={registrar[:30]}")
            return result
        
        except requests.exceptions.Timeout:
            print(f"[WHOIS] ❌ TIMEOUT for {domain} (>{self.config.request_timeout}s)")
            return None
        except requests.exceptions.RequestException as e:
            print(f"[WHOIS] ❌ HTTP ERROR for {domain}: {e}")
            return None
        except Exception as e:
            print(f"[WHOIS] ❌ PARSE ERROR for {domain}: {e}")
            return None
    
    def _estimate_domain_age(self, domain: str) -> float:
        """
        Estimate domain age in days. Uses WHOIS API if available,
        otherwise falls back to heuristics.
        """
        # Try WHOIS first
        whois_data = self._whois_lookup(domain)
        if whois_data and whois_data.get('age_days', -1) >= 0:
            return whois_data['age_days']
        
        # Fallback heuristics
        if re.search(r'\d{4,}', domain):
            return 30.0
        
        if domain.count('-') > 2:
            return 60.0
        
        known_legitimate = ['google.com', 'microsoft.com', 'amazon.com', 
                           'apple.com', 'linkedin.com', 'indeed.com']
        if any(legit in domain for legit in known_legitimate):
            return 3650.0
        
        return 365.0
    
    def _check_email_domain(self, domain: Optional[str]) -> Tuple[int, int]:
        """
        Check if email domain is corporate or free provider
        Returns: (is_corporate, is_free_provider)
        """
        if not domain:
            return 0, 0
        
        domain = domain.lower()
        
        is_free = int(domain in self.FREE_EMAIL_PROVIDERS)
        is_corporate = int(not is_free and '.' in domain)
        
        return is_corporate, is_free
    
    def _estimate_linkedin_presence(self, company_name: str, domain: Optional[str]) -> Tuple[int, float, float]:
        """
        Estimate LinkedIn presence (in production, use LinkedIn API)
        Returns: (has_page, employee_count, followers)
        """
        # Placeholder - would use LinkedIn API in production
        # Returns probabilistic estimates based on company indicators
        
        if not company_name:
            return 0, 0.0, 0.0
        
        company_name = str(company_name).lower()
        
        # Large company indicators
        large_indicators = ['inc.', 'corp.', 'corporation', 'ltd.', 'limited', 
                           'llc', 'group', 'international', 'global']
        
        is_large = any(ind in company_name for ind in large_indicators)
        
        if is_large:
            return 1, 1000.0, 5000.0
        
        # Medium company
        if domain and not any(tld in domain for tld in ['.tk', '.ml']):
            return 1, 50.0, 200.0
        
        # Unknown/small
        return 0, 0.0, 0.0
    
    def _calculate_website_score(self, company_profile: str) -> Tuple[int, int, float]:
        """
        Calculate website quality score based on profile text
        Returns: (has_contact, has_career, professional_score)
        """
        if pd.isna(company_profile):
            return 0, 0, 0.0
        
        text = str(company_profile).lower()
        
        # Check for contact/career page mentions
        has_contact = int(any(word in text for word in ['contact us', 'contact@', 'support@']))
        has_career = int(any(word in text for word in ['careers', 'jobs', 'join us', 'hiring']))
        
        # Professional indicators
        professional_words = ['founded', 'headquarters', 'employees', 'revenue', 
                             'industry', 'mission', 'values', 'team']
        professional_count = sum(1 for word in professional_words if word in text)
        
        # Red flags
        red_flags = ['urgent', 'immediate start', 'no experience', 'work from home', 
                    'quick money', 'guaranteed', 'act now']
        red_flag_count = sum(1 for flag in red_flags if flag in text)
        
        # Calculate score
        score = min(1.0, (professional_count * 0.15) - (red_flag_count * 0.2) + 0.3)
        score = max(0.0, score)  # Clamp to [0, 1]
        
        return has_contact, has_career, score
    
    def _check_known_fake_patterns(self, company_profile: str, domain: Optional[str]) -> int:
        """Check for known fake job patterns"""
        text = str(company_profile).lower()
        
        # Check domain
        if domain:
            for pattern in self.KNOWN_FAKE_PATTERNS:
                if pattern in domain:
                    return 1
        
        # Check text for suspicious phrases
        suspicious_phrases = [
            'send money', 'pay upfront', 'registration fee', 'training fee',
            'buy starter kit', 'investment required', 'pay to work'
        ]
        
        return int(any(phrase in text for phrase in suspicious_phrases))
    
    def extract(self, company_profile: str, contact_email: Optional[str] = None,
                company_name: Optional[str] = None, source_url: Optional[str] = None) -> NetworkFeatures:
        """
        Extract network features for a job posting
        
        Args:
            company_profile: Company description/profile text
            contact_email: Contact email from job posting
            company_name: Company name
            source_url: URL of the page the job was scraped from
            
        Returns:
            NetworkFeatures object with extracted features
        """
        # Check cache (include source_url so different URLs get different cache entries)
        cache_key = self._get_cache_key(f"{company_profile}_{source_url}", contact_email)
        
        if cache_key in self._cache:
            features, timestamp = self._cache[cache_key]
            if self._is_cache_valid(timestamp):
                # Invalidate stale entries that were cached before WHOIS integration
                if features.whois_domain_age_days == -1.0 and self.whois_api_key:
                    print(f"[CACHE] ♻ Invalidating stale entry (no WHOIS data) for {cache_key}")
                else:
                    print(f"[CACHE] ✓ Cache hit for {cache_key}")
                    logger.debug(f"Cache hit for {cache_key}")
                    return features
        
        logger.debug(f"Extracting features for {cache_key}")
        
        # Extract domain info
        domain = self._extract_domain_from_text(company_profile)
        email_domain = self._extract_email_domain(contact_email)
        
        # Fallback: extract domain from source URL (e.g. the job posting page)
        if not domain and source_url:
            try:
                parsed = urlparse(source_url)
                if parsed.hostname:
                    # Strip 'www.' prefix
                    domain = parsed.hostname.lower().replace('www.', '')
            except Exception:
                pass
        
        print(f"\n[EXTRACT] Domain: {domain or 'None'} | Email domain: {email_domain or 'None'} | Source URL: {source_url or 'None'}")
        
        # Initialize features
        features = NetworkFeatures()
        
        # Domain features
        if domain:
            features.domain_age_days = self._estimate_domain_age(domain)
            features.domain_has_ssl = self._check_ssl(domain)
            features.domain_suspicious_tld = self._check_suspicious_tld(domain)
        
        # Email features
        if email_domain:
            features.email_is_corporate, features.email_is_free_provider = \
                self._check_email_domain(email_domain)
            features.email_domain_age_days = self._estimate_domain_age(email_domain)
        
        # LinkedIn presence
        features.has_linkedin_page, features.linkedin_employee_count, features.linkedin_followers = \
            self._estimate_linkedin_presence(company_name or "", domain)
        
        # Website quality
        features.website_has_contact_page, features.website_has_career_page, \
            features.website_professional_score = self._calculate_website_score(company_profile)
        
        # Known fake detection
        features.is_known_fake_domain = self._check_known_fake_patterns(company_profile, domain)
        
        # Calculate trust score
        trust_score = 0.5  # Neutral
        
        # WHOIS-derived signals (if available)
        whois_data = None
        if domain:
            whois_data = self._whois_lookup(domain)
        
        if whois_data and whois_data.get('age_days', -1) >= 0:
            age = whois_data['age_days']
            features.whois_domain_age_days = age
            features.whois_registrar = whois_data.get('registrar', '')
            features.whois_expires_soon = whois_data.get('expires_soon', 0)
            
            # Age-based trust: older domains more trustworthy
            if age > 1825:      # 5+ years
                trust_score += 0.15
            elif age > 365:     # 1-5 years
                trust_score += 0.08
            elif age > 90:      # 3-12 months
                trust_score += 0.0
            elif age > 30:      # 1-3 months
                trust_score -= 0.1
            else:               # less than 30 days
                trust_score -= 0.2
            
            # Expiring soon = suspicious
            if whois_data.get('expires_soon'):
                trust_score -= 0.1
        else:
            # No WHOIS data, use basic heuristics
            pass
        
        if features.domain_has_ssl:
            trust_score += 0.1
        if features.domain_suspicious_tld:
            trust_score -= 0.2
        if features.email_is_corporate:
            trust_score += 0.1
        if features.email_is_free_provider:
            trust_score -= 0.1
        if features.has_linkedin_page:
            trust_score += 0.1
        if features.is_known_fake_domain:
            trust_score -= 0.3
        if features.website_professional_score > 0.6:
            trust_score += 0.1
        
        features.domain_trust_score = max(0.0, min(1.0, trust_score))
        
        # Cache result
        self._cache[cache_key] = (features, time.time())
        self._save_cache()
        
        return features
    
    def extract_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract features for entire dataframe
        
        Args:
            df: DataFrame with columns: company_profile, contact_email, company_name
            
        Returns:
            Numpy array of shape (n_samples, n_features)
        """
        features_list = []
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                logger.info(f"Processed {idx}/{len(df)} rows")
            
            features = self.extract(
                company_profile=row.get('company_profile', ''),
                contact_email=row.get('contact_email'),
                company_name=row.get('company_name')
            )
            features_list.append(features.to_vector())
        
        return np.array(features_list)
    
    def __del__(self):
        """Cleanup - save cache on deletion"""
        try:
            self._save_cache()
        except Exception:
            pass  # Builtins may be gone during interpreter shutdown
