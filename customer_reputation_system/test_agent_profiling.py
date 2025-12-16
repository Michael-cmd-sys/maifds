#!/usr/bin/env python3
"""
Test script for agent risk profiling with synthetic data
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import json
from maifds_governance.storage.database import DatabaseManager
from maifds_governance.synthetic_data.generator import SyntheticDataGenerator
from pathlib import Path

def test_agent_risk_profiling():
    """Test agent risk profiling with synthetic data"""
    
    # Initialize database
    db_path = Path("test_agent_risk.db")
    db = DatabaseManager(db_path)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    generator = SyntheticDataGenerator(seed=42)
    dataset = generator.generate_synthetic_dataset(
        num_agents=10,
        num_merchants=50,
        num_mules=10,
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
    
    # Test agent risk calculation
    print("\nTesting agent risk calculation...")
    
    # Import here to avoid path issues
    from maifds_governance.agents.calculator import AgentRiskCalculator
    
    calculator = AgentRiskCalculator(db)
    
    # Test with first agent
    test_agent = dataset["agents"][0]
    agent_id = test_agent["agent_id"]
    
    print(f"\nAnalyzing agent: {test_agent['agent_name']} ({agent_id})")
    print(f"Original risk score: {test_agent['risk_score']:.3f}")
    
    try:
        # Calculate risk factors
        risk_factors = calculator.calculate_agent_risk_factors(agent_id)
        print(f"Risk factors:")
        print(f"  Recruitment velocity: {risk_factors.recruitment_velocity:.3f}")
        print(f"  Network growth rate: {risk_factors.network_growth_rate:.3f}")
        print(f"  Transaction anomaly: {risk_factors.transaction_anomaly_score:.3f}")
        print(f"  Geographic dispersion: {risk_factors.geographic_dispersion:.3f}")
        print(f"  Temporal patterns: {risk_factors.temporal_patterns:.3f}")
        print(f"  Communication risk: {risk_factors.communication_risk:.3f}")
        print(f"  Financial behavior: {risk_factors.financial_behavior_score:.3f}")
        print(f"  Association risk: {risk_factors.association_risk:.3f}")
        
        # Calculate composite risk
        composite_risk = risk_factors.calculate_composite_risk()
        print(f"Composite risk score: {composite_risk:.3f}")
        
        # Update agent profile
        updated_profile = calculator.update_agent_risk_profile(agent_id)
        print(f"Updated risk score: {updated_profile.risk_score:.3f}")
        print(f"Updated credibility score: {updated_profile.credibility_score:.3f}")
        
    except Exception as e:
        print(f"Error calculating risk: {e}")
        import traceback
        traceback.print_exc()
    
    # Get high risk agents
    print("\nHigh risk agents (threshold 0.7):")
    high_risk_agents = calculator.get_high_risk_agents(0.7)
    for agent in high_risk_agents:
        print(f"  - {agent.agent_name}: {agent.risk_score:.3f}")
    
    # Cleanup
    if db_path.exists():
        db_path.unlink()
        print(f"\nCleaned up test database: {db_path}")

if __name__ == "__main__":
    test_agent_risk_profiling()