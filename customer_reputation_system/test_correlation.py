#!/usr/bin/env python3
"""
Test script for cross-entity correlation engine
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import json
from customer_reputation_system.src.storage.database import DatabaseManager
from customer_reputation_system.src.synthetic_data.generator import SyntheticDataGenerator
from pathlib import Path

def test_cross_entity_correlation():
    """Test cross-entity correlation with synthetic data"""
    
    # Initialize database
    db_path = Path("test_correlation.db")
    db = DatabaseManager(db_path)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    generator = SyntheticDataGenerator(seed=42)
    dataset = generator.generate_synthetic_dataset(
        num_agents=15,
        num_merchants=60,
        num_mules=20,
        high_risk_ratio=0.3
    )
    
    # Load data into database (simplified version)
    print("Loading data into database...")
    # print(dataset)
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
        # print("Is this running...")
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
        # print("Certainly this then right?")
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
    
    print(f"Loaded {len(dataset['agents'])} agents, {len(dataset['merchants'])} merchants, "
          f"{len(dataset['relationships'])} relationships")
    
    # Test cross-entity correlation
    print("\nTesting cross-entity correlation...")
    
    # Import here to avoid path issues
    from src.correlation.engine import CrossEntityCorrelationEngine
    
    correlation_engine = CrossEntityCorrelationEngine(db)
    
    # Test agent-merchant correlations
    print("\nFinding agent-merchant correlations...")
    correlations = correlation_engine.find_agent_merchant_correlations(min_correlation_score=0.6)
    
    print(f"Found {len(correlations)} high-correlation agent-merchant pairs:")
    for corr in correlations[:5]:  # Show top 5
        print(f"  - {corr['agent_name']} <-> {corr['merchant_name']}: "
              f"{corr['correlation_score']:.3f} ({corr['transaction_count']} tx, "
              f"${corr['total_amount']:,.2f})")
    
    # Test money laundering chains
    print("\nDetecting money laundering chains...")
    chains = correlation_engine.detect_money_laundering_chains(min_chain_length=3)
    
    print(f"Found {len(chains)} suspicious chains:")
    for chain in chains[:3]:  # Show top 3
        print(f"  - Chain length {chain['length']}: {chain['suspiciousness_score']:.3f} "
              f"suspiciousness, ${chain['total_amount']:,.2f}, {chain['chain_type']}")
    
    # Test network clusters
    print("\nAnalyzing network clusters...")
    clusters = correlation_engine.analyze_network_clusters()
    
    print(f"Analyzed {len(clusters)} network clusters:")
    for cluster in clusters[:3]:  # Show top 3
        print(f"  - Network {cluster['network_id']}: {cluster['cluster_risk_score']:.3f} risk, "
              f"{cluster['agent_count']} agents, {cluster['merchant_count']} merchants, "
              f"{cluster['mule_count']} mules ({cluster['cluster_size']} size)")
    
    # Test high-value targets
    print("\nFinding high-value targets...")
    targets = correlation_engine.find_high_value_targets(min_transaction_amount=5000)
    
    print(f"Found {len(targets)} high-value targets:")
    for target in targets[:5]:  # Show top 5
        print(f"  - {target['agent_name']} -> {target['merchant_name']}: "
              f"${target['total_transaction_amount']:,.2f} ({target['transaction_count']} tx)")
    
    # Test cross-entity funds flow tracing
    if dataset["agents"]:
        test_agent = dataset["agents"][0]["agent_id"]
        print(f"\nTracing cross-entity funds flow from: {test_agent}")
        
        try:
            flow_analysis = correlation_engine.trace_cross_entity_funds_flow(test_agent, max_depth=3)
            print(f"Funds flow analysis:")
            print(f"  - Entity type: {flow_analysis['entity_type']}")
            print(f"  - Max depth reached: {flow_analysis['max_depth_reached']}")
            print(f"  - Flow paths found: {len(flow_analysis['flow_paths'])}")
            print(f"  - Suspicious patterns: {len(flow_analysis['suspicious_patterns'])}")
            
            # Show entity summary
            for entity_type, summary in flow_analysis['entity_summary'].items():
                if summary['count'] > 0:
                    print(f"    {entity_type}: {summary['count']} entities, "
                          f"${summary['amount']:,.2f}")
            
            # Show suspicious patterns
            for pattern in flow_analysis['suspicious_patterns'][:3]:
                print(f"    Suspicious: {pattern['pattern_type']} - "
                      f"{pattern['from_entity']} -> {pattern['to_entity']}")
            
        except Exception as e:
            print(f"Error tracing funds flow: {e}")
            import traceback
            traceback.print_exc()
    
    # Cleanup
    if db_path.exists():
        db_path.unlink()
        print(f"\nCleaned up test database: {db_path}")

if __name__ == "__main__":
    test_cross_entity_correlation()