#!/usr/bin/env python3
"""
Test script for mule network detection with synthetic data
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import json
from customer_reputation_system.src.storage.database import DatabaseManager
from customer_reputation_system.src.synthetic_data.generator import SyntheticDataGenerator
from pathlib import Path

def test_mule_network_detection():
    """Test mule network detection with synthetic data"""
    
    # Initialize database
    db_path = Path("test_mule_detection.db")
    db = DatabaseManager(db_path)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    generator = SyntheticDataGenerator(seed=42)
    dataset = generator.generate_synthetic_dataset(
        num_agents=10,
        num_merchants=50,
        num_mules=15,
        high_risk_ratio=0.3
    )
    
    # Load data into database
    print("Loading data into database...")
    
    # Load agents
    for agent_data in dataset["agents"]:
        query = """
        INSERT OR REPLACE INTO agents 
        (agent_id, agent_name, credibility_score, risk_score, total_recruits,
         active_merchants, network_depth, recruitment_rate, avg_transaction_amount,
         suspicious_activity_count, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        db.execute_update(query, (
            agent_data["agent_id"],
            agent_data["agent_name"],
            agent_data["credibility_score"],
            agent_data["risk_score"],
            agent_data["total_recruits"],
            agent_data["active_merchants"],
            agent_data["network_depth"],
            agent_data["recruitment_rate"],
            agent_data["avg_transaction_amount"],
            agent_data["suspicious_activity_count"],
            agent_data["created_at"],
            agent_data["updated_at"]
        ))
    
    # Load merchants
    for merchant_data in dataset["merchants"]:
        query = """
        INSERT OR REPLACE INTO merchants 
        (merchant_id, merchant_name, total_reports, average_rating, reputation_score, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        db.execute_update(query, (
            merchant_data["merchant_id"],
            merchant_data["merchant_name"],
            merchant_data["total_reports"],
            merchant_data["average_rating"],
            merchant_data["reputation_score"],
            merchant_data["created_at"],
            merchant_data["updated_at"]
        ))
    
    # Load relationships
    for rel_data in dataset["relationships"]:
        query = """
        INSERT OR REPLACE INTO agent_networks 
        (network_id, agent_id, merchant_id, relationship_type, strength_score,
         transaction_count, total_amount, risk_level, first_interaction, last_interaction, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        db.execute_update(query, (
            rel_data["network_id"],
            rel_data["agent_id"],
            rel_data["merchant_id"],
            rel_data["relationship_type"],
            rel_data["strength_score"],
            rel_data["transaction_count"],
            rel_data["total_amount"],
            rel_data["risk_level"],
            rel_data["first_interaction"],
            rel_data["last_interaction"],
            rel_data["created_at"]
        ))
    
    # Load mule accounts
    for mule_data in dataset["mule_accounts"]:
        query = """
        INSERT OR REPLACE INTO mule_accounts 
        (account_id, account_type, mule_score, network_id, transaction_patterns,
         risk_indicators, is_confirmed_mule, detection_date, rapid_transaction_count,
         circular_transaction_count, avg_hold_time_minutes, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        db.execute_update(query, (
            mule_data["account_id"],
            mule_data["account_type"],
            mule_data["mule_score"],
            mule_data["network_id"],
            mule_data["transaction_patterns"],
            mule_data["risk_indicators"],
            mule_data["is_confirmed_mule"],
            mule_data["detection_date"],
            mule_data["rapid_transaction_count"],
            mule_data["circular_transaction_count"],
            mule_data["avg_hold_time_minutes"],
            mule_data["created_at"],
            mule_data["updated_at"]
        ))
    
    print(f"Loaded {len(dataset['agents'])} agents, {len(dataset['merchants'])} merchants, "
          f"{len(dataset['relationships'])} relationships, {len(dataset['mule_accounts'])} mule accounts")
    
    # Test mule network detection
    print("\nTesting mule network detection...")
    
    # Import here to avoid path issues
    from src.mule_network.detector import MuleNetworkDetector
    
    detector = MuleNetworkDetector(db)
    
    # Test mule detection
    print("\nDetecting mule accounts...")
    detected_mules = detector.detect_mule_accounts(threshold=0.6)
    
    print(f"Detected {len(detected_mules)} potential mule accounts:")
    for mule in detected_mules[:5]:  # Show top 5
        print(f"  - {mule.account_id} ({mule.account_type}): {mule.mule_score:.3f}")
    
    # Test suspicious patterns
    print("\nFinding suspicious patterns...")
    patterns = detector.find_suspicious_patterns()
    
    for pattern_type, pattern_list in patterns.items():
        print(f"\n{pattern_type.replace('_', ' ').title()}: {len(pattern_list)} patterns")
        for pattern in pattern_list[:3]:  # Show top 3
            if isinstance(pattern, dict):
                key = next(iter(pattern.keys()))
                print(f"  - {key}: {pattern[key]}")
    
    # Test network analysis
    if dataset["relationships"]:
        test_network_id = dataset["relationships"][0]["network_id"]
        print(f"\nAnalyzing network: {test_network_id}")
        
        try:
            network_metrics = detector.analyze_network_structure(test_network_id)
            print(f"Network metrics:")
            print(f"  - Total nodes: {network_metrics.total_nodes}")
            print(f"  - Mule density: {network_metrics.mule_density:.3f}")
            print(f"  - Transaction velocity: {network_metrics.transaction_velocity:.2f} tx/day")
            print(f"  - Network risk score: {network_metrics.network_risk_score:.3f}")
            print(f"  - Centralization index: {network_metrics.centralization_index:.3f}")
        except Exception as e:
            print(f"Error analyzing network: {e}")
    
    # Test money flow tracing
    if dataset["mule_accounts"]:
        test_account = dataset["mule_accounts"][0]["account_id"]
        print(f"\nTracing money flow from: {test_account}")
        
        try:
            flow_trace = detector.trace_money_flow(test_account, depth=2)
            print(f"Money flow analysis:")
            print(f"  - Accounts reached: {flow_trace['total_accounts_reached']}")
            print(f"  - Total amount flowed: ${flow_trace['total_amount_flowed']:,.2f}")
            print(f"  - Flow paths found: {len(flow_trace['flow_paths'])}")
            
            # Show top flow paths
            for path in flow_trace['flow_paths'][:3]:
                print(f"    {path['from_account']} -> {path['to_account']}: "
                      f"${path['amount']:,.2f} ({path['transaction_count']} tx)")
        except Exception as e:
            print(f"Error tracing money flow: {e}")
    
    # Cleanup
    if db_path.exists():
        db_path.unlink()
        print(f"\nCleaned up test database: {db_path}")

if __name__ == "__main__":
    test_mule_network_detection()