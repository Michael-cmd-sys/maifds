"""
Domain Intelligence Module for MAIFDS
Integrates URL expansion, WHOIS data, and threat intelligence from WhoisXML API
"""

import requests
from datetime import datetime
from typing import Dict, Optional, Any
import os
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class DomainIntelligence:
    """
    Multi-layer domain intelligence gathering for phishing detection
    """
    
    def __init__(self, whoisxml_api_key: Optional[str] = None):
        """
        Initialize Domain Intelligence with WhoisXML API credentials
        
        Args:
            whoisxml_api_key: API key for WhoisXML services
        """
        self.api_key = whoisxml_api_key or os.getenv('WHOISXML_API_KEY')
        self.whois_base_url = "https://www.whoisxmlapi.com/whoisserver/WhoisService"
        self.reputation_base_url = "https://domain-reputation.whoisxmlapi.com/api/v2"
        self.geo_base_url = "https://ip-geolocation.whoisxmlapi.com/api/v1"
        self.dns_base_url = "https://www.whoisxmlapi.com/whoisserver/DNSService"
        
    def expand_url(self, url: str, max_redirects: int = 10) -> Dict[str, Any]:
        """
        Follow URL redirects to reveal the final destination domain
        Handles URL shorteners like bit.ly, tinyurl, etc.
        
        Args:
            url: The URL to expand (can be shortened)
            max_redirects: Maximum number of redirects to follow
            
        Returns:
            Dict with original_url, final_url, redirect_chain, and final_domain
        """
        try:
            # Ensure URL has a scheme
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                
            redirect_chain = [url]
            
            # Follow redirects manually
            session = requests.Session()
            session.max_redirects = max_redirects
            
            response = session.head(url, allow_redirects=True, timeout=5)
            final_url = response.url
            
            # Extract final domain
            parsed = urlparse(final_url)
            final_domain = parsed.netloc
            
            # Build redirect chain if we were redirected
            if final_url != url:
                redirect_chain.append(final_url)
            
            return {
                'success': True,
                'original_url': url,
                'final_url': final_url,
                'final_domain': final_domain,
                'redirect_chain': redirect_chain,
                'was_shortened': len(redirect_chain) > 1
            }
            
        except requests.RequestException as e:
            logger.error(f"URL expansion failed for {url}: {e}")
            # If expansion fails, try to extract domain from original URL
            try:
                parsed = urlparse(url if url.startswith('http') else 'https://' + url)
                domain = parsed.netloc
                return {
                    'success': False,
                    'error': str(e),
                    'original_url': url,
                    'final_url': url,
                    'final_domain': domain,
                    'redirect_chain': [url],
                    'was_shortened': False
                }
            except Exception as parse_error:
                return {
                    'success': False,
                    'error': f"URL expansion and parsing failed: {e}, {parse_error}",
                    'original_url': url,
                    'final_url': None,
                    'final_domain': None
                }
    
    def get_whois_data(self, domain: str) -> Dict[str, Any]:
        """
        Fetch WHOIS data for a domain including age and registrant info
        
        Args:
            domain: Domain name to look up
            
        Returns:
            Dict with creation_date, age_days, registrant, registrar, and raw data
        """
        if not self.api_key:
            return {'error': 'WHOISXML_API_KEY not configured', 'success': False}
        
        try:
            params = {
                'apiKey': self.api_key,
                'domainName': domain,
                'outputFormat': 'JSON'
            }
            
            response = requests.get(self.whois_base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Extract relevant fields
            whois_record = data.get('WhoisRecord', {})
            created_date_raw = whois_record.get('createdDate')
            registrant = whois_record.get('registrant', {})
            registrar = whois_record.get('registrarName')
            
            # Calculate age
            age_days = None
            creation_date = None
            if created_date_raw:
                try:
                    # WhoisXML returns ISO format like "1997-09-15T07:00:00+0000" or "1997-09-15T04:00:00Z"
                    # Python's fromisoformat needs +00:00 format (with colon), so fix it
                    date_str = created_date_raw.replace('Z', '+00:00')
                    # Handle +0000 format (add colon to make +00:00)
                    if '+' in date_str and date_str[-4:].isdigit():
                        date_str = date_str[:-2] + ':' + date_str[-2:]
                    creation_date = datetime.fromisoformat(date_str)
                    age_days = (datetime.now(creation_date.tzinfo) - creation_date).days
                except Exception as date_error:
                    logger.warning(f"Date parsing failed for {domain}: {date_error} (raw: {created_date_raw})")
            
            return {
                'success': True,
                'domain': domain,
                'creation_date': created_date_raw,
                'age_days': age_days,
                'registrant_name': registrant.get('name', 'Redacted for Privacy'),
                'registrant_org': registrant.get('organization', 'N/A'),
                'registrant_country': registrant.get('country', 'N/A'),
                'registrar': registrar,
                'raw_data': whois_record
            }
            
        except requests.RequestException as e:
            logger.error(f"WHOIS lookup failed for {domain}: {e}")
            return {'success': False, 'error': str(e), 'domain': domain}
    
    def check_reputation(self, domain: str) -> Dict[str, Any]:
        """
        Check domain reputation using WhoisXML Domain Reputation API
        
        Args:
            domain: Domain to check
            
        Returns:
            Dict with reputation_score, mode (safe/suspicious/malicious), and details
        """
        if not self.api_key:
            return {'error': 'WHOISXML_API_KEY not configured', 'success': False}
        
        try:
            params = {
                'apiKey': self.api_key,
                'domainName': domain
            }
            
            response = requests.get(self.reputation_base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Extract reputation metrics
            test_results = data.get('testResults', {})
            reputation_score = data.get('reputationScore', 0)  # 0-100, higher is better
            mode = data.get('mode', 'unknown')  # "safe", "suspicious", "malicious"
            
            return {
                'success': True,
                'domain': domain,
                'reputation_score': reputation_score,
                'mode': mode,
                'is_malicious': mode == 'malicious',
                'is_suspicious': mode in ['suspicious', 'malicious'],
                'test_results': test_results
            }
            
        except requests.RequestException as e:
            logger.error(f"Reputation check failed for {domain}: {e}")
            return {'success': False, 'error': str(e), 'domain': domain}
    
    def get_geolocation(self, domain: str) -> Dict[str, Any]:
        """
        Get IP geolocation for the domain's hosting server
        
        Args:
            domain: Domain to geolocate
            
        Returns:
            Dict with country, city, ISP, and coordinates
        """
        if not self.api_key:
            return {'error': 'WHOISXML_API_KEY not configured', 'success': False}
        
        try:
            # First resolve domain to IP (we'll use the DNS service for this)
            # For simplicity, we can also just pass the domain directly
            params = {
                'apiKey': self.api_key,
                'domain': domain
            }
            
            response = requests.get(self.geo_base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            location = data.get('location', {})
            
            return {
                'success': True,
                'domain': domain,
                'country': location.get('country', 'Unknown'),
                'country_code': location.get('countryCode', 'XX'),
                'city': location.get('city', 'Unknown'),
                'isp': data.get('isp', 'Unknown'),
                'latitude': location.get('lat'),
                'longitude': location.get('lng'),
                'postal_code': location.get('postalCode'),
                'timezone': location.get('timezone')
            }
            
        except requests.RequestException as e:
            logger.error(f"Geolocation lookup failed for {domain}: {e}")
            return {'success': False, 'error': str(e), 'domain': domain}
    
    def check_dns_records(self, domain: str) -> Dict[str, Any]:
        """
        Check DNS records including MX (email) records
        
        Args:
            domain: Domain to check
            
        Returns:
            Dict with MX records, A records, and other DNS data
        """
        if not self.api_key:
            return {'error': 'WHOISXML_API_KEY not configured', 'success': False}
        
        try:
            params = {
                'apiKey': self.api_key,
                'domainName': domain,
                'type': 'mx',  # Get MX records
                'outputFormat': 'JSON'
            }
            
            response = requests.get(self.dns_base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            mx_records = data.get('DNSData', {}).get('dnsRecords', [])
            
            return {
                'success': True,
                'domain': domain,
                'has_mx_records': len(mx_records) > 0,
                'mx_count': len(mx_records),
                'mx_records': mx_records,
                'dns_data': data
            }
            
        except requests.RequestException as e:
            logger.error(f"DNS lookup failed for {domain}: {e}")
            return {'success': False, 'error': str(e), 'domain': domain}
    
    def analyze_url_comprehensive(self, url: str) -> Dict[str, Any]:
        """
        Comprehensive analysis combining all intelligence sources
        
        Args:
            url: URL to analyze (can be shortened)
            
        Returns:
            Dict with all intelligence data and risk assessment
        """
        results = {
            'original_url': url,
            'timestamp': datetime.now().isoformat()
        }
        
        # Step 1: Expand URL to get real domain
        expansion = self.expand_url(url)
        results['expansion'] = expansion
        
        if not expansion.get('final_domain'):
            results['risk_assessment'] = 'HIGH_RISK'
            results['risk_reason'] = 'Unable to resolve domain'
            return results
        
        domain = expansion['final_domain']
        
        # Step 2: Get WHOIS data (domain age, owner)
        whois_data = self.get_whois_data(domain)
        results['whois'] = whois_data
        
        # Step 3: Check reputation
        reputation = self.check_reputation(domain)
        results['reputation'] = reputation
        
        # Step 4: Get geolocation
        geolocation = self.get_geolocation(domain)
        results['geolocation'] = geolocation
        
        # Step 5: Check DNS/MX records
        dns = self.check_dns_records(domain)
        results['dns'] = dns
        
        # Risk Assessment
        risk_signals = []
        risk_level = 'LOW_RISK'
        
        # Check domain age
        age_days = whois_data.get('age_days')
        if age_days is not None:
            if age_days < 30:
                risk_signals.append('Domain less than 30 days old')
                risk_level = 'HIGH_RISK'
            elif age_days < 90:
                risk_signals.append('Domain less than 90 days old')
                risk_level = 'MEDIUM_RISK' if risk_level == 'LOW_RISK' else risk_level
        
        # Check reputation
        if reputation.get('is_malicious'):
            risk_signals.append('Domain flagged as malicious')
            risk_level = 'HIGH_RISK'
        elif reputation.get('is_suspicious'):
            risk_signals.append('Domain flagged as suspicious')
            risk_level = 'MEDIUM_RISK' if risk_level == 'LOW_RISK' else risk_level
        
        # Check for URL shortening (hiding real domain)
        if expansion.get('was_shortened'):
            risk_signals.append('URL was shortened (hiding real domain)')
            risk_level = 'MEDIUM_RISK' if risk_level == 'LOW_RISK' else risk_level
        
        # Check for missing email infrastructure
        if not dns.get('has_mx_records'):
            risk_signals.append('No email server configured (suspicious for business sites)')
            risk_level = 'MEDIUM_RISK' if risk_level == 'LOW_RISK' else risk_level
        
        results['risk_assessment'] = {
            'level': risk_level,
            'signals': risk_signals,
            'signal_count': len(risk_signals)
        }
        
        return results


def get_domain_intelligence_instance() -> DomainIntelligence:
    """
    Factory function to get a configured DomainIntelligence instance
    """
    return DomainIntelligence()
