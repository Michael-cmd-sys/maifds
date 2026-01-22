#!/usr/bin/env python3
"""
Test script for Domain Intelligence features
Run this to verify your WhoisXML API integration
"""

import requests
import json

API_BASE = "http://localhost:5000"

def test_expand_url():
    """Test URL expansion"""
    print("\n=== Testing URL Expansion ===")
    
    response = requests.post(f"{API_BASE}/expand-url", json={
        "url": "https://google.com"
    })
    
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

def test_whois():
    """Test WHOIS lookup"""
    print("\n=== Testing WHOIS Lookup ===")
    
    response = requests.post(f"{API_BASE}/domain-whois", json={
        "domain": "google.com"
    })
    
    print(f"Status: {response.status_code}")
    data = response.json()
    
    if data.get('success'):
        print(f"Domain: {data['domain']}")
        print(f"Age: {data.get('age_days', 'Unknown')} days")
        print(f"Created: {data.get('creation_date', 'Unknown')}")
        print(f"Registrar: {data.get('registrar', 'Unknown')}")
    else:
        print(f"Error: {data.get('error')}")

def test_reputation():
    """Test reputation check"""
    print("\n=== Testing Reputation Check ===")
    
    response = requests.post(f"{API_BASE}/domain-reputation", json={
        "domain": "google.com"
    })
    
    print(f"Status: {response.status_code}")
    data = response.json()
    
    if data.get('success'):
        print(f"Domain: {data['domain']}")
        print(f"Score: {data.get('reputation_score', 'Unknown')}")
        print(f"Mode: {data.get('mode', 'Unknown')}")
        print(f"Malicious: {data.get('is_malicious', False)}")
    else:
        print(f"Error: {data.get('error')}")

def test_comprehensive():
    """Test comprehensive domain analysis"""
    print("\n=== Testing Comprehensive Analysis ===")
    
    response = requests.post(f"{API_BASE}/analyze-domain", json={
        "url": "https://google.com"
    })
    
    print(f"Status: {response.status_code}")
    data = response.json()
    
    print(f"Original URL: {data.get('original_url')}")
    
    if 'whois' in data and data['whois'].get('success'):
        print(f"Domain Age: {data['whois'].get('age_days')} days")
    
    if 'reputation' in data and data['reputation'].get('success'):
        print(f"Reputation: {data['reputation'].get('mode')} (score: {data['reputation'].get('reputation_score')})")
    
    if 'geolocation' in data and data['geolocation'].get('success'):
        print(f"Location: {data['geolocation'].get('country')}")
    
    if 'risk_assessment' in data:
        risk = data['risk_assessment']
        print(f"\nRisk Level: {risk.get('level')}")
        print(f"Signals: {risk.get('signal_count')}")
        for signal in risk.get('signals', []):
            print(f"  - {signal}")

if __name__ == "__main__":
    print("Domain Intelligence API Test Suite")
    print("=" * 50)
    print("Make sure the API is running: python api_mindspore.py")
    print("=" * 50)
    
    try:
        test_expand_url()
        test_whois()
        test_reputation()
        test_comprehensive()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed!")
        print("Check the output above for any errors.")
        print("=" * 50)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to API")
        print("Make sure the API is running on http://localhost:5000")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
