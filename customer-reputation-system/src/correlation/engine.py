"""
Cross-entity correlation engine for agent/merchant/mule network analysis
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict, Counter
from customer_reputation_system.src.agents.models import AgentRiskProfile
from customer_reputation_system.src.mule_network.models import MuleAccount, NetworkRiskMetrics
from customer_reputation_system.src.storage.database import DatabaseManager


class CrossEntityCorrelationEngine:
    """Analyze correlations between agents, merchants, and mule accounts"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def find_agent_merchant_correlations(self, min_correlation_score: float = 0.5) -> List[Dict[str, Any]]:
        """Find suspicious correlations between agents and merchants"""
        
        correlations = []
        
        # Get all agent-merchant relationships
        query = """
        SELECT 
            a.agent_id, a.agent_name, a.risk_score as agent_risk_score,
            m.merchant_id, m.merchant_name, m.reputation_score as merchant_reputation,
            an.network_id, an.relationship_type, an.strength_score,
            an.transaction_count, an.total_amount, an.risk_level,
            an.first_interaction, an.last_interaction
        FROM agents a
        JOIN agent_networks an ON a.agent_id = an.agent_id
        JOIN merchants m ON an.merchant_id = m.merchant_id
        ORDER BY an.transaction_count DESC, an.total_amount DESC
        """
        
        relationships = self.db.execute_query(query)
        
        for rel in relationships:
            # Calculate correlation score
            correlation_score = self._calculate_agent_merchant_correlation(rel)
            
            if correlation_score >= min_correlation_score:
                correlations.append({
                    "agent_id": rel["agent_id"],
                    "agent_name": rel["agent_name"],
                    "merchant_id": rel["merchant_id"],
                    "merchant_name": rel["merchant_name"],
                    "correlation_score": correlation_score,
                    "relationship_type": rel["relationship_type"],
                    "strength_score": rel["strength_score"],
                    "transaction_count": rel["transaction_count"],
                    "total_amount": rel["total_amount"],
                    "risk_level": rel["risk_level"],
                    "agent_risk_score": rel["agent_risk_score"],
                    "merchant_reputation_score": rel["merchant_reputation"],
                    "relationship_duration_days": self._calculate_relationship_duration(
                        rel["first_interaction"], rel["last_interaction"]
                    )
                })
        
        return correlations

    def detect_money_laundering_chains(self, min_chain_length: int = 3) -> List[Dict[str, Any]]:
        """Detect potential money laundering chains"""
        
        # Build transaction graph
        graph = self._build_comprehensive_graph()
        chains = []
        
        # Find chains using DFS
        visited = set()
        for start_node in graph:
            if start_node not in visited:
                self._find_chains_from_node(
                    graph, start_node, [], chains, min_chain_length, visited
                )
        
        # Score chains based on suspiciousness
        scored_chains = []
        for chain in chains:
            chain_score = self._score_chain_suspiciousness(chain, graph)
            if chain_score > 0.5:  # Threshold for suspicious chains
                scored_chains.append({
                    "chain": chain,
                    "length": len(chain),
                    "suspiciousness_score": chain_score,
                    "total_amount": self._calculate_chain_amount(chain, graph),
                    "chain_type": self._classify_chain_type(chain, graph)
                })
        
        return sorted(scored_chains, key=lambda x: x["suspiciousness_score"], reverse=True)

    def analyze_network_clusters(self) -> List[Dict[str, Any]]:
        """Analyze network clusters for suspicious patterns"""
        
        # Get network data
        networks = self._get_all_networks()
        clusters = []
        
        for network in networks:
            cluster_analysis = self._analyze_single_network_cluster(network)
            clusters.append(cluster_analysis)
        
        # Sort by risk score
        return sorted(clusters, key=lambda x: x["cluster_risk_score"], reverse=True)

    def find_high_value_targets(self, min_transaction_amount: float = 10000) -> List[Dict[str, Any]]:
        """Find high-value targets for fraud prevention"""
        
        query = """
        SELECT 
            an.agent_id, a.agent_name, a.risk_score,
            an.merchant_id, m.merchant_name, m.reputation_score,
            SUM(an.total_amount) as total_transaction_amount,
            COUNT(*) as transaction_count,
            AVG(an.total_amount) as avg_transaction_amount,
            MAX(an.total_amount) as max_transaction_amount,
            an.risk_level
        FROM agents a
        JOIN agent_networks an ON a.agent_id = an.agent_id
        JOIN merchants m ON an.merchant_id = m.merchant_id
        WHERE an.total_amount >= ?
        GROUP BY an.agent_id, an.merchant_id
        HAVING total_transaction_amount >= ?
        ORDER BY total_transaction_amount DESC
        LIMIT 100
        """
        
        targets = self.db.execute_query(query, (min_transaction_amount, min_transaction_amount))
        
        # Enrich with additional risk factors
        enriched_targets = []
        for target in targets:
            enrichment = self._enrich_target_data(target)
            enriched_targets.append(enrichment)
        
        return enriched_targets

    def trace_cross_entity_funds_flow(self, account_id: str, max_depth: int = 4) -> Dict[str, Any]:
        """Trace funds flow across different entity types"""
        
        # Build comprehensive graph
        graph = self._build_comprehensive_graph()
        
        # Perform BFS with entity type tracking
        flow_analysis = {
            "source_account": account_id,
            "entity_type": self._get_entity_type(account_id),
            "flow_paths": [],
            "entity_summary": defaultdict(lambda: {"count": 0, "amount": 0.0}),
            "suspicious_patterns": [],
            "max_depth_reached": 0
        }
        
        visited = set()
        queue = [(account_id, 0, [], self._get_entity_type(account_id))]
        
        while queue and len(flow_analysis["flow_paths"]) < 200:
            current_account, depth, path, entity_type = queue.pop(0)
            
            if depth > max_depth or current_account in visited:
                continue
            
            visited.add(current_account)
            current_path = path + [current_account]
            flow_analysis["max_depth_reached"] = max(flow_analysis["max_depth_reached"], depth)
            
            # Update entity summary
            flow_analysis["entity_summary"][entity_type]["count"] += 1
            
            # Explore neighbors
            if current_account in graph:
                for neighbor, edge_data in graph[current_account].items():
                    if neighbor not in visited:
                        neighbor_entity_type = self._get_entity_type(neighbor)
                        
                        # Update amounts
                        amount = edge_data.get("total_amount", 0)
                        flow_analysis["entity_summary"][entity_type]["amount"] += amount
                        flow_analysis["entity_summary"][neighbor_entity_type]["amount"] += amount
                        
                        # Add flow path
                        flow_analysis["flow_paths"].append({
                            "from_account": current_account,
                            "to_account": neighbor,
                            "from_entity_type": entity_type,
                            "to_entity_type": neighbor_entity_type,
                            "amount": amount,
                            "transaction_count": edge_data.get("transaction_count", 0),
                            "depth": depth + 1,
                            "risk_level": edge_data.get("risk_level", "medium")
                        })
                        
                        # Check for suspicious patterns
                        if self._is_suspicious_cross_entity_flow(entity_type, neighbor_entity_type, edge_data):
                            flow_analysis["suspicious_patterns"].append({
                                "pattern_type": "cross_entity_suspicious",
                                "from_entity": entity_type,
                                "to_entity": neighbor_entity_type,
                                "details": edge_data
                            })
                        
                        queue.append((neighbor, depth + 1, current_path, neighbor_entity_type))
        
        return flow_analysis

    def _calculate_agent_merchant_correlation(self, relationship: Dict[str, Any]) -> float:
        """Calculate correlation score between agent and merchant"""
        
        agent_risk = relationship["agent_risk_score"]
        merchant_reputation = 1.0 - relationship["merchant_reputation_score"]  # Invert for risk
        transaction_volume = min(1.0, relationship["total_amount"] / 100000.0)
        transaction_frequency = min(1.0, relationship["transaction_count"] / 100.0)
        relationship_strength = relationship["strength_score"]
        risk_level_multiplier = {"low": 0.5, "medium": 1.0, "high": 1.5}.get(relationship["risk_level"], 1.0)
        
        # Combine factors
        correlation = (
            (agent_risk * 0.25) +
            (merchant_reputation * 0.20) +
            (transaction_volume * 0.20) +
            (transaction_frequency * 0.15) +
            (relationship_strength * 0.10) +
            (risk_level_multiplier * 0.10)
        )
        
        return min(1.0, correlation)

    def _build_comprehensive_graph(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Build comprehensive transaction graph with all entities"""
        
        graph = defaultdict(dict)
        
        # Add agent-merchant relationships
        query = """
        SELECT agent_id, merchant_id, transaction_count, total_amount, 
               strength_score, risk_level, first_interaction, last_interaction
        FROM agent_networks
        """
        
        for row in self.db.execute_query(query):
            agent_id = row["agent_id"]
            merchant_id = row["merchant_id"]
            
            # Add bidirectional edges
            graph[agent_id][merchant_id] = {
                "transaction_count": row["transaction_count"],
                "total_amount": row["total_amount"],
                "strength_score": row["strength_score"],
                "risk_level": row["risk_level"],
                "relationship_type": "agent_merchant",
                "first_interaction": row["first_interaction"],
                "last_interaction": row["last_interaction"]
            }
            
            graph[merchant_id][agent_id] = {
                "transaction_count": row["transaction_count"],
                "total_amount": row["total_amount"],
                "strength_score": row["strength_score"],
                "risk_level": row["risk_level"],
                "relationship_type": "merchant_agent",
                "first_interaction": row["first_interaction"],
                "last_interaction": row["last_interaction"]
            }
        
        return dict(graph)

    def _find_chains_from_node(self, graph: Dict[str, Dict[str, Dict[str, Any]]], 
                              node: str, current_chain: List[str], 
                              chains: List[List[str]], min_length: int, 
                              visited: Set[str]):
        """Find chains starting from a node using DFS"""
        
        if node in current_chain:  # Found a cycle
            cycle_start = current_chain.index(node)
            chain = current_chain[cycle_start:] + [node]
            if len(chain) >= min_length:
                chains.append(chain)
            return
        
        if len(current_chain) >= min_length + 2:  # Limit chain length
            chains.append(current_chain + [node])
            return
        
        visited.add(node)
        current_chain.append(node)
        
        if node in graph:
            for neighbor in graph[node]:
                if neighbor not in current_chain:  # Avoid immediate cycles
                    self._find_chains_from_node(
                        graph, neighbor, current_chain, chains, min_length, visited
                    )
        
        current_chain.pop()

    def _score_chain_suspiciousness(self, chain: List[str], 
                                  graph: Dict[str, Dict[str, Dict[str, Any]]]) -> float:
        """Score how suspicious a transaction chain is"""
        
        if len(chain) < 3:
            return 0.0
        
        suspiciousness = 0.0
        
        for i in range(len(chain) - 1):
            from_node = chain[i]
            to_node = chain[i + 1]
            
            if from_node in graph and to_node in graph[from_node]:
                edge_data = graph[from_node][to_node]
                
                # Factors for suspiciousness
                high_amount = min(1.0, edge_data.get("total_amount", 0) / 50000.0)
                high_frequency = min(1.0, edge_data.get("transaction_count", 0) / 50.0)
                high_risk = 1.0 if edge_data.get("risk_level") == "high" else 0.5
                high_strength = edge_data.get("strength_score", 0)
                
                suspiciousness += (high_amount + high_frequency + high_risk + high_strength) / 4.0
        
        return min(1.0, suspiciousness / (len(chain) - 1))

    def _calculate_chain_amount(self, chain: List[str], 
                            graph: Dict[str, Dict[str, Dict[str, Any]]]) -> float:
        """Calculate total amount in a chain"""
        
        total_amount = 0.0
        
        for i in range(len(chain) - 1):
            from_node = chain[i]
            to_node = chain[i + 1]
            
            if from_node in graph and to_node in graph[from_node]:
                total_amount += graph[from_node][to_node].get("total_amount", 0)
        
        return total_amount

    def _classify_chain_type(self, chain: List[str], 
                           graph: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
        """Classify the type of transaction chain"""
        
        entity_types = [self._get_entity_type(node) for node in chain]
        type_counts = Counter(entity_types)
        
        if len(type_counts) == 1:
            return f"{entity_types[0]}_only"
        elif "agent" in type_counts and "merchant" in type_counts:
            return "agent_merchant_mixed"
        elif "mule" in type_counts:
            return "mule_involved"
        else:
            return "mixed_entities"

    def _get_all_networks(self) -> List[Dict[str, Any]]:
        """Get all networks with their entities"""
        
        query = """
        SELECT DISTINCT network_id,
               COUNT(DISTINCT agent_id) as agent_count,
               COUNT(DISTINCT merchant_id) as merchant_count,
               COUNT(*) as total_relationships,
               SUM(transaction_count) as total_transactions,
               SUM(total_amount) as total_amount,
               AVG(strength_score) as avg_strength
        FROM agent_networks
        GROUP BY network_id
        ORDER BY total_amount DESC
        """
        
        return self.db.execute_query(query)

    def _analyze_single_network_cluster(self, network: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single network cluster"""
        
        network_id = network["network_id"]
        
        # Get mule accounts in this network
        mule_query = """
        SELECT COUNT(*) as mule_count, AVG(mule_score) as avg_mule_score
        FROM mule_accounts 
        WHERE network_id = ?
        """
        mule_data = self.db.execute_query(mule_query, (network_id,))
        mule_info = mule_data[0] if mule_data else {"mule_count": 0, "avg_mule_score": 0}
        
        # Calculate cluster risk score
        agent_ratio = network["agent_count"] / max(1, network["agent_count"] + network["merchant_count"])
        mule_ratio = mule_info["mule_count"] / max(1, network["total_relationships"])
        avg_transaction_size = network["total_amount"] / max(1, network["total_transactions"])
        
        cluster_risk = (
            (agent_ratio * 0.3) +
            (mule_ratio * 0.4) +
            (min(1.0, avg_transaction_size / 10000.0) * 0.2) +
            (network["avg_strength"] * 0.1)
        )
        
        return {
            "network_id": network_id,
            "agent_count": network["agent_count"],
            "merchant_count": network["merchant_count"],
            "total_relationships": network["total_relationships"],
            "mule_count": mule_info["mule_count"],
            "avg_mule_score": mule_info["avg_mule_score"] or 0,
            "total_transactions": network["total_transactions"],
            "total_amount": network["total_amount"],
            "avg_strength": network["avg_strength"],
            "cluster_risk_score": cluster_risk,
            "cluster_size": "small" if network["total_relationships"] < 10 else "medium" if network["total_relationships"] < 50 else "large"
        }

    def _enrich_target_data(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich target data with additional risk factors"""
        
        agent_id = target["agent_id"]
        merchant_id = target["merchant_id"]
        
        # Get agent's other relationships
        agent_query = """
        SELECT COUNT(*) as other_relationships, AVG(risk_score) as avg_agent_network_risk
        FROM agents a
        JOIN agent_networks an ON a.agent_id = an.agent_id
        WHERE a.agent_id = ? AND an.merchant_id != ?
        """
        agent_enrichment = self.db.execute_query(agent_query, (agent_id, merchant_id))
        
        # Get merchant's other relationships
        merchant_query = """
        SELECT COUNT(*) as other_relationships, AVG(risk_score) as avg_merchant_network_risk
        FROM merchants m
        JOIN agent_networks an ON m.merchant_id = an.merchant_id
        WHERE m.merchant_id = ? AND an.agent_id != ?
        """
        merchant_enrichment = self.db.execute_query(merchant_query, (merchant_id, agent_id))
        
        enrichment = target.copy()
        enrichment.update({
            "agent_other_relationships": agent_enrichment[0]["other_relationships"] if agent_enrichment else 0,
            "agent_network_avg_risk": agent_enrichment[0]["avg_agent_network_risk"] if agent_enrichment else 0,
            "merchant_other_relationships": merchant_enrichment[0]["other_relationships"] if merchant_enrichment else 0,
            "merchant_network_avg_risk": merchant_enrichment[0]["avg_merchant_network_risk"] if merchant_enrichment else 0,
            "risk_concentration_score": self._calculate_risk_concentration(target, agent_enrichment, merchant_enrichment)
        })
        
        return enrichment

    def _calculate_risk_concentration(self, target: Dict[str, Any], 
                                  agent_enrichment: List[Dict[str, Any]], 
                                  merchant_enrichment: List[Dict[str, Any]]) -> float:
        """Calculate risk concentration score"""
        
        # High concentration if most activity is with one partner
        total_agent_rels = agent_enrichment[0]["other_relationships"] + 1 if agent_enrichment else 1
        total_merchant_rels = merchant_enrichment[0]["other_relationships"] + 1 if merchant_enrichment else 1
        
        agent_concentration = 1.0 / total_agent_rels if total_agent_rels > 0 else 0
        merchant_concentration = 1.0 / total_merchant_rels if total_merchant_rels > 0 else 0
        
        return (agent_concentration + merchant_concentration) / 2.0

    def _get_entity_type(self, entity_id: str) -> str:
        """Determine entity type from ID"""
        if entity_id.startswith("agent_"):
            return "agent"
        elif entity_id.startswith("merchant_"):
            return "merchant"
        elif entity_id.startswith("account_"):
            return "mule"
        else:
            return "unknown"

    def _calculate_relationship_duration(self, first_interaction: str, last_interaction: str) -> int:
        """Calculate relationship duration in days"""
        if not first_interaction or not last_interaction:
            return 0
        
        try:
            first_date = datetime.fromisoformat(first_interaction)
            last_date = datetime.fromisoformat(last_interaction)
            return (last_date - first_date).days
        except:
            return 0

    def _is_suspicious_cross_entity_flow(self, from_entity: str, to_entity: str, 
                                       edge_data: Dict[str, Any]) -> bool:
        """Check if cross-entity flow is suspicious"""
        
        # Define suspicious patterns
        suspicious_patterns = [
            ("agent", "mule"),  # Agent to mule
            ("mule", "merchant"),  # Mule to merchant
            ("mule", "mule"),  # Mule to mule
        ]
        
        # Check for suspicious entity type combinations
        if (from_entity, to_entity) in suspicious_patterns:
            return True
        
        # Check for high-risk transaction characteristics
        high_amount = edge_data.get("total_amount", 0) > 50000
        high_frequency = edge_data.get("transaction_count", 0) > 100
        high_risk_level = edge_data.get("risk_level") == "high"
        
        return high_amount or high_frequency or high_risk_level