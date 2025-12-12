"""
Synthetic data generator for agent/merchant risk profiling and mule network detection
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
import json


class SyntheticDataGenerator:
    """Generate synthetic data for testing agent/merchant risk profiling and mule detection"""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.agent_names = [
            "John Smith", "Maria Garcia", "Li Wei", "Ahmed Hassan", "Sarah Johnson",
            "Robert Chen", "Fatima Al-Rashid", "David Kim", "Priya Patel", "Mohammed Ali"
        ]
        self.merchant_names = [
            "QuickMart Store", "Digital Solutions Inc", "Global Trading Co", "Tech Services Ltd",
            "Retail Express", "Online Marketplace", "Wholesale Distributors", "Local Shop",
            "E-Commerce Hub", "Business Solutions"
        ]
        self.cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", 
                      "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"]

    def generate_agent(self, is_high_risk: bool = False) -> Dict[str, Any]:
        """Generate a synthetic agent profile"""
        agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        agent_name = random.choice(self.agent_names)
        
        if is_high_risk:
            credibility_score = random.uniform(0.1, 0.4)
            risk_score = random.uniform(0.7, 0.95)
            total_recruits = random.randint(15, 50)
            active_merchants = random.randint(10, 40)
            network_depth = random.randint(3, 6)
            recruitment_rate = random.uniform(2.0, 8.0)
            avg_transaction_amount = random.uniform(5000, 25000)
            suspicious_activity_count = random.randint(5, 20)
        else:
            credibility_score = random.uniform(0.6, 0.95)
            risk_score = random.uniform(0.05, 0.3)
            total_recruits = random.randint(1, 10)
            active_merchants = random.randint(1, 8)
            network_depth = random.randint(1, 3)
            recruitment_rate = random.uniform(0.1, 1.5)
            avg_transaction_amount = random.uniform(500, 5000)
            suspicious_activity_count = random.randint(0, 3)

        return {
            "agent_id": agent_id,
            "agent_name": agent_name,
            "credibility_score": credibility_score,
            "risk_score": risk_score,
            "total_recruits": total_recruits,
            "active_merchants": active_merchants,
            "network_depth": network_depth,
            "recruitment_rate": recruitment_rate,
            "avg_transaction_amount": avg_transaction_amount,
            "suspicious_activity_count": suspicious_activity_count,
            "created_at": (datetime.now() - timedelta(days=random.randint(30, 365))).isoformat(),
            "updated_at": datetime.now().isoformat()
        }

    def generate_merchant(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate a synthetic merchant profile"""
        merchant_id = f"merchant_{uuid.uuid4().hex[:8]}"
        merchant_name = random.choice(self.merchant_names)
        
        total_reports = random.randint(0, 20)
        average_rating = random.uniform(1.0, 5.0) if total_reports > 0 else None
        reputation_score = random.uniform(0.2, 0.9)

        return {
            "merchant_id": merchant_id,
            "merchant_name": merchant_name,
            "total_reports": total_reports,
            "average_rating": average_rating,
            "reputation_score": reputation_score,
            "created_at": (datetime.now() - timedelta(days=random.randint(15, 300))).isoformat(),
            "updated_at": datetime.now().isoformat()
        }

    def generate_network_relationship(self, agent_id: str, merchant_id: str, 
                                     is_high_risk: bool = False) -> Dict[str, Any]:
        """Generate a network relationship between agent and merchant"""
        network_id = f"network_{uuid.uuid4().hex[:8]}"
        relationship_types = ["recruited", "associated", "transactional", "managed"]
        
        if is_high_risk:
            strength_score = random.uniform(0.7, 1.0)
            transaction_count = random.randint(50, 500)
            total_amount = random.uniform(50000, 500000)
            risk_level = random.choice(["medium", "high"])
        else:
            strength_score = random.uniform(0.1, 0.6)
            transaction_count = random.randint(1, 50)
            total_amount = random.uniform(1000, 50000)
            risk_level = random.choice(["low", "medium"])

        first_interaction = datetime.now() - timedelta(days=random.randint(30, 180))
        last_interaction = first_interaction + timedelta(days=random.randint(1, 90))

        return {
            "network_id": network_id,
            "agent_id": agent_id,
            "merchant_id": merchant_id,
            "relationship_type": random.choice(relationship_types),
            "strength_score": strength_score,
            "transaction_count": transaction_count,
            "total_amount": total_amount,
            "risk_level": risk_level,
            "first_interaction": first_interaction.isoformat(),
            "last_interaction": last_interaction.isoformat(),
            "created_at": datetime.now().isoformat()
        }

    def generate_mule_account(self, network_id: Optional[str] = None, is_confirmed_mule: bool = False) -> Dict[str, Any]:
        """Generate a synthetic mule account"""
        account_id = f"account_{uuid.uuid4().hex[:8]}"
        account_types = ["merchant", "agent", "individual"]
        
        if is_confirmed_mule:
            mule_score = random.uniform(0.8, 1.0)
            rapid_transaction_count = random.randint(20, 100)
            circular_transaction_count = random.randint(5, 30)
            avg_hold_time_minutes = random.uniform(5, 60)
        else:
            mule_score = random.uniform(0.0, 0.7)
            rapid_transaction_count = random.randint(0, 20)
            circular_transaction_count = random.randint(0, 5)
            avg_hold_time_minutes = random.uniform(60, 1440)

        transaction_patterns = {
            "peak_hours": [random.randint(9, 17) for _ in range(random.randint(1, 3))],
            "avg_interval_hours": random.uniform(0.5, 24),
            "weekend_activity": random.choice([True, False])
        }

        risk_indicators = {
            "unusual_amounts": random.choice([True, False]),
            "rapid_succession": rapid_transaction_count > 10,
            "circular_patterns": circular_transaction_count > 0,
            "short_hold_times": avg_hold_time_minutes < 30
        }

        return {
            "account_id": account_id,
            "account_type": random.choice(account_types),
            "mule_score": mule_score,
            "network_id": network_id,
            "transaction_patterns": json.dumps(transaction_patterns),
            "risk_indicators": json.dumps(risk_indicators),
            "is_confirmed_mule": is_confirmed_mule,
            "detection_date": datetime.now().isoformat() if is_confirmed_mule else None,
            "rapid_transaction_count": rapid_transaction_count,
            "circular_transaction_count": circular_transaction_count,
            "avg_hold_time_minutes": avg_hold_time_minutes,
            "created_at": (datetime.now() - timedelta(days=random.randint(1, 90))).isoformat(),
            "updated_at": datetime.now().isoformat()
        }

    def generate_fraud_report(self, reporter_id: str, merchant_id: str, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate a synthetic fraud report"""
        report_id = f"report_{uuid.uuid4().hex[:8]}"
        report_types = ["fraud", "suspicious_activity", "money_laundering", "account_takeover"]
        
        titles = [
            "Suspicious transaction pattern detected",
            "Unusual account behavior",
            "Potential money laundering activity",
            "Fraudulent transaction attempt",
            "Account compromise suspected"
        ]
        
        descriptions = [
            "Multiple rapid transactions detected in short time period",
            "Circular payment patterns observed between related accounts",
            "Unusually large transactions followed by quick fund transfers",
            "Account shows signs of unauthorized access and manipulation",
            "Transaction patterns consistent with money mule activity"
        ]

        return {
            "report_id": report_id,
            "timestamp": (datetime.now() - timedelta(hours=random.randint(1, 72))).isoformat(),
            "reporter_id": reporter_id,
            "merchant_id": merchant_id,
            "report_type": random.choice(report_types),
            "rating": random.randint(1, 5),
            "title": random.choice(titles),
            "description": random.choice(descriptions),
            "transaction_id": f"tx_{uuid.uuid4().hex[:8]}",
            "amount": random.uniform(100, 50000),
            "metadata_json": json.dumps({"agent_id": agent_id} if agent_id else {}),
            "created_at": datetime.now().isoformat()
        }

    def generate_synthetic_dataset(self, num_agents: int = 50, num_merchants: int = 200, 
                                  num_mules: int = 30, high_risk_ratio: float = 0.2) -> Dict[str, List[Dict]]:
        """Generate a complete synthetic dataset"""
        
        agents = []
        merchants = []
        relationships = []
        mule_accounts = []
        reports = []
        
        # Generate agents
        high_risk_agents = int(num_agents * high_risk_ratio)
        for i in range(num_agents):
            is_high_risk = i < high_risk_agents
            agent = self.generate_agent(is_high_risk)
            agents.append(agent)
        
        # Generate merchants
        for _ in range(num_merchants):
            merchant = self.generate_merchant()
            merchants.append(merchant)
        
        # Generate network relationships
        for agent in agents:
            # Each agent recruits multiple merchants
            num_recruits = random.randint(1, min(8, len(merchants)))
            recruited_merchants = random.sample(merchants, num_recruits)
            
            for merchant in recruited_merchants:
                relationship = self.generate_network_relationship(
                    agent["agent_id"], 
                    merchant["merchant_id"],
                    agent["risk_score"] > 0.7
                )
                relationships.append(relationship)
        
        # Generate mule accounts
        confirmed_mules = int(num_mules * 0.3)
        for i in range(num_mules):
            is_confirmed = i < confirmed_mules
            network_id = random.choice(relationships)["network_id"] if relationships else None
            mule = self.generate_mule_account(network_id, is_confirmed)
            mule_accounts.append(mule)
        
        # Generate fraud reports
        for _ in range(num_merchants):
            merchant = random.choice(merchants)
            reporter_id = f"reporter_{uuid.uuid4().hex[:8]}"
            agent = random.choice(agents) if random.random() < 0.3 else None
            report = self.generate_fraud_report(
                reporter_id, 
                merchant["merchant_id"],
                agent["agent_id"] if agent else None
            )
            reports.append(report)
        
        return {
            "agents": agents,
            "merchants": merchants,
            "relationships": relationships,
            "mule_accounts": mule_accounts,
            "reports": reports
        }

    def save_dataset_to_files(self, dataset: Dict[str, List[Dict]], output_dir: str = "synthetic_data"):
        """Save synthetic dataset to JSON files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for table_name, data in dataset.items():
            filename = f"{output_dir}/{table_name}.json"
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved {len(data)} records to {filename}")


if __name__ == "__main__":
    generator = SyntheticDataGenerator(seed=42)
    dataset = generator.generate_synthetic_dataset(
        num_agents=50,
        num_merchants=200,
        num_mules=30,
        high_risk_ratio=0.2
    )
    
    generator.save_dataset_to_files(dataset, "customer-reputation-system/data/synthetic")
    
    print(f"\nDataset Summary:")
    print(f"Agents: {len(dataset['agents'])}")
    print(f"Merchants: {len(dataset['merchants'])}")
    print(f"Relationships: {len(dataset['relationships'])}")
    print(f"Mule Accounts: {len(dataset['mule_accounts'])}")
    print(f"Reports: {len(dataset['reports'])}")