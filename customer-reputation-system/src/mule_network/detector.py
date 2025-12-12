"""
Mule network detection algorithm with graph analysis
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict, deque
from .models import MuleAccount, MuleRiskFactors, NetworkRiskMetrics, NetworkRelationship
from ..storage.database import DatabaseManager


class MuleNetworkDetector:
    """Detect mule accounts and analyze money laundering networks"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def detect_mule_accounts(self, threshold: float = 0.7) -> List[MuleAccount]:
        """Detect potential mule accounts based on risk factors"""
        
        # Get all accounts from database
        accounts = self._get_all_accounts()
        detected_mules = []
        
        for account in accounts:
            # Calculate mule risk factors
            risk_factors = self._calculate_mule_risk_factors(account["account_id"])
            mule_probability = risk_factors.calculate_mule_probability()
            
            # Update account with new mule score
            if mule_probability >= threshold:
                mule_account = MuleAccount(
                    account_id=account["account_id"],
                    account_type=account["account_type"],
                    mule_score=mule_probability,
                    network_id=account.get("network_id"),
                    transaction_patterns=account.get("transaction_patterns"),
                    risk_indicators=account.get("risk_indicators"),
                    is_confirmed_mule=account.get("is_confirmed_mule", False),
                    detection_date=datetime.fromisoformat(account["detection_date"]) if account.get("detection_date") else None,
                    rapid_transaction_count=account.get("rapid_transaction_count", 0),
                    circular_transaction_count=account.get("circular_transaction_count", 0),
                    avg_hold_time_minutes=account.get("avg_hold_time_minutes", 0.0),
                    created_at=datetime.fromisoformat(account["created_at"]) if account.get("created_at") else None,
                    updated_at=datetime.fromisoformat(account["updated_at"]) if account.get("updated_at") else None
                )
                detected_mules.append(mule_account)
                
                # Update database
                self._update_mule_score(account["account_id"], mule_probability)
        
        return detected_mules

    def analyze_network_structure(self, network_id: str) -> NetworkRiskMetrics:
        """Analyze the structure and risk of a money laundering network"""
        
        # Get network data
        network_data = self._get_network_data(network_id)
        if not network_data:
            raise ValueError(f"Network {network_id} not found")
        
        # Calculate network metrics
        total_nodes = len(network_data["nodes"])
        mule_accounts = [node for node in network_data["nodes"] if node.get("mule_score", 0) > 0.7]
        mule_density = len(mule_accounts) / total_nodes if total_nodes > 0 else 0.0
        
        # Calculate transaction velocity
        total_transactions = sum(edge.get("transaction_count", 0) for edge in network_data["edges"])
        network_age_days = self._calculate_network_age(network_data)
        transaction_velocity = total_transactions / max(1, network_age_days)
        
        # Calculate average transaction amount
        total_amount = sum(edge.get("total_amount", 0) for edge in network_data["edges"])
        avg_transaction_amount = total_amount / max(1, len(network_data["edges"]))
        
        # Calculate network risk score
        network_risk_score = self._calculate_network_risk_score(network_data)
        
        # Calculate centralization index
        centralization_index = self._calculate_centralization_index(network_data)
        
        # Count communities (simplified)
        community_count = self._detect_communities(network_data)
        
        # Count bridge edges
        bridge_edges_count = self._count_bridge_edges(network_data)
        
        return NetworkRiskMetrics(
            network_id=network_id,
            total_nodes=total_nodes,
            mule_density=mule_density,
            transaction_velocity=transaction_velocity,
            avg_transaction_amount=avg_transaction_amount,
            network_risk_score=network_risk_score,
            centralization_index=centralization_index,
            community_count=community_count,
            bridge_edges_count=bridge_edges_count,
            last_updated=datetime.now()
        )

    def find_suspicious_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Find suspicious transaction patterns in the network"""
        
        patterns = {
            "circular_transactions": self._find_circular_transactions(),
            "rapid_succession": self._find_rapid_succession_patterns(),
            "layered_transactions": self._find_layered_transactions(),
            "unusual_timing": self._find_unusual_timing_patterns(),
            "amount_anomalies": self._find_amount_anomalies()
        }
        
        return patterns

    def trace_money_flow(self, account_id: str, depth: int = 3) -> Dict[str, Any]:
        """Trace money flow from a specific account"""
        
        # Build transaction graph
        graph = self._build_transaction_graph()
        
        # Perform BFS to trace money flow
        flow_path = []
        visited = set()
        queue = deque([(account_id, 0, [])])
        
        while queue and len(flow_path) < 100:  # Limit results
            current_account, current_depth, path = queue.popleft()
            
            if current_depth > depth or current_account in visited:
                continue
                
            visited.add(current_account)
            current_path = path + [current_account]
            
            # Get outgoing transactions
            if current_account in graph:
                for neighbor, edge_data in graph[current_account].items():
                    if neighbor not in visited:
                        flow_path.append({
                            "from_account": current_account,
                            "to_account": neighbor,
                            "amount": edge_data.get("total_amount", 0),
                            "transaction_count": edge_data.get("transaction_count", 0),
                            "depth": current_depth + 1,
                            "path": current_path + [neighbor]
                        })
                        queue.append((neighbor, current_depth + 1, current_path))
        
        return {
            "source_account": account_id,
            "max_depth": depth,
            "flow_paths": flow_path,
            "total_accounts_reached": len(visited),
            "total_amount_flowed": sum(path["amount"] for path in flow_path)
        }

    def _get_all_accounts(self) -> List[Dict[str, Any]]:
        """Get all accounts from database"""
        query = """
        SELECT account_id, account_type, mule_score, network_id, transaction_patterns,
               risk_indicators, is_confirmed_mule, detection_date, rapid_transaction_count,
               circular_transaction_count, avg_hold_time_minutes, created_at, updated_at
        FROM mule_accounts
        """
        return self.db.execute_query(query)

    def _calculate_mule_risk_factors(self, account_id: str) -> MuleRiskFactors:
        """Calculate mule risk factors for an account"""
        
        # Get account data
        account_data = self._get_account_data(account_id)
        if not account_data:
            return MuleRiskFactors()
        
        # Get transaction patterns
        transaction_patterns = json.loads(account_data.get("transaction_patterns", "{}"))
        risk_indicators = json.loads(account_data.get("risk_indicators", "{}"))
        
        # Calculate individual risk factors
        rapid_transaction_score = min(1.0, account_data.get("rapid_transaction_count", 0) / 50.0)
        circular_transaction_score = min(1.0, account_data.get("circular_transaction_count", 0) / 10.0)
        short_hold_time_score = 1.0 - min(1.0, account_data.get("avg_hold_time_minutes", 1440) / 1440.0)
        network_centrality_score = self._calculate_account_centrality(account_id)
        amount_anomaly_score = self._calculate_amount_anomaly_score(account_id)
        temporal_pattern_score = self._calculate_temporal_pattern_score(transaction_patterns)
        geographic_anomaly_score = self._calculate_geographic_anomaly_score(account_id)
        behavioral_consistency_score = 1.0 - risk_indicators.get("unusual_amounts", 0)
        
        return MuleRiskFactors(
            rapid_transaction_score=rapid_transaction_score,
            circular_transaction_score=circular_transaction_score,
            short_hold_time_score=short_hold_time_score,
            network_centrality_score=network_centrality_score,
            amount_anomaly_score=amount_anomaly_score,
            temporal_pattern_score=temporal_pattern_score,
            geographic_anomaly_score=geographic_anomaly_score,
            behavioral_consistency_score=behavioral_consistency_score
        )

    def _get_network_data(self, network_id: str) -> Optional[Dict[str, Any]]:
        """Get network data including nodes and edges"""
        
        # Get nodes (accounts in this network)
        nodes_query = """
        SELECT account_id, account_type, mule_score, rapid_transaction_count,
               circular_transaction_count, avg_hold_time_minutes
        FROM mule_accounts 
        WHERE network_id = ?
        """
        nodes = self.db.execute_query(nodes_query, (network_id,))
        
        # Get edges (relationships)
        edges_query = """
        SELECT agent_id, merchant_id, transaction_count, total_amount, strength_score
        FROM agent_networks 
        WHERE network_id = ?
        """
        edges = self.db.execute_query(edges_query, (network_id,))
        
        if not nodes and not edges:
            return None
        
        return {
            "network_id": network_id,
            "nodes": nodes,
            "edges": edges
        }

    def _find_circular_transactions(self) -> List[Dict[str, Any]]:
        """Find circular transaction patterns"""
        
        # Build transaction graph
        graph = self._build_transaction_graph()
        circular_patterns = []
        
        # Detect cycles using DFS
        visited = set()
        for start_node in graph:
            if start_node not in visited:
                path = []
                self._dfs_cycles(graph, start_node, visited, path, circular_patterns)
        
        return circular_patterns[:50]  # Limit results

    def _find_rapid_succession_patterns(self) -> List[Dict[str, Any]]:
        """Find rapid succession transaction patterns"""
        
        query = """
        SELECT account_id, rapid_transaction_count, avg_hold_time_minutes
        FROM mule_accounts 
        WHERE rapid_transaction_count > 10 OR avg_hold_time_minutes < 60
        ORDER BY rapid_transaction_count DESC
        LIMIT 50
        """
        
        return self.db.execute_query(query)

    def _find_layered_transactions(self) -> List[Dict[str, Any]]:
        """Find layered transaction patterns (multiple hops)"""
        
        # This is a simplified implementation
        # In practice, this would analyze transaction chains
        query = """
        SELECT an.network_id, COUNT(*) as layer_count, SUM(an.transaction_count) as total_transactions
        FROM agent_networks an
        GROUP BY an.network_id
        HAVING layer_count > 5
        ORDER BY layer_count DESC
        LIMIT 50
        """
        
        return self.db.execute_query(query)

    def _find_unusual_timing_patterns(self) -> List[Dict[str, Any]]:
        """Find unusual timing patterns"""
        
        # Simplified timing pattern detection
        query = """
        SELECT account_id, transaction_patterns
        FROM mule_accounts 
        WHERE transaction_patterns IS NOT NULL
        LIMIT 50
        """
        
        results = []
        for row in self.db.execute_query(query):
            patterns = json.loads(row.get("transaction_patterns", "{}"))
            if patterns.get("weekend_activity", False):
                results.append(row)
        
        return results

    def _find_amount_anomalies(self) -> List[Dict[str, Any]]:
        """Find amount anomalies"""
        
        query = """
        SELECT account_id, network_id, rapid_transaction_count, circular_transaction_count
        FROM mule_accounts 
        WHERE rapid_transaction_count > 20 OR circular_transaction_count > 5
        ORDER BY rapid_transaction_count DESC, circular_transaction_count DESC
        LIMIT 50
        """
        
        return self.db.execute_query(query)

    def _build_transaction_graph(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Build transaction graph from network data"""
        
        graph = defaultdict(dict)
        
        query = """
        SELECT agent_id, merchant_id, transaction_count, total_amount, strength_score
        FROM agent_networks
        """
        
        for row in self.db.execute_query(query):
            agent_id = row["agent_id"]
            merchant_id = row["merchant_id"]
            
            graph[agent_id][merchant_id] = {
                "transaction_count": row["transaction_count"],
                "total_amount": row["total_amount"],
                "strength_score": row["strength_score"]
            }
        
        return dict(graph)

    def _dfs_cycles(self, graph: Dict[str, Dict[str, Dict[str, Any]]], 
                   node: str, visited: Set[str], path: List[str], 
                   cycles: List[Dict[str, Any]]):
        """DFS to detect cycles in transaction graph"""
        
        if node in path:
            # Found a cycle
            cycle_start = path.index(node)
            cycle = path[cycle_start:] + [node]
            
            cycles.append({
                "cycle": cycle,
                "length": len(cycle) - 1,
                "accounts_involved": len(cycle) - 1
            })
            return
        
        if node in visited:
            return
        
        visited.add(node)
        path.append(node)
        
        if node in graph:
            for neighbor in graph[node]:
                self._dfs_cycles(graph, neighbor, visited, path, cycles)
        
        path.pop()

    def _calculate_network_age(self, network_data: Dict[str, Any]) -> float:
        """Calculate network age in days"""
        # Simplified - assume 30 days average age
        return 30.0

    def _calculate_network_risk_score(self, network_data: Dict[str, Any]) -> float:
        """Calculate overall network risk score"""
        
        if not network_data["nodes"]:
            return 0.0
        
        # Average mule score across nodes
        total_mule_score = sum(node.get("mule_score", 0) for node in network_data["nodes"])
        avg_mule_score = total_mule_score / len(network_data["nodes"])
        
        # Factor in transaction volume
        total_transactions = sum(edge.get("transaction_count", 0) for edge in network_data["edges"])
        transaction_factor = min(1.0, total_transactions / 1000.0)
        
        # Combine factors
        return (avg_mule_score * 0.7) + (transaction_factor * 0.3)

    def _calculate_centralization_index(self, network_data: Dict[str, Any]) -> float:
        """Calculate network centralization index"""
        
        if not network_data["edges"]:
            return 0.0
        
        # Simplified centralization based on degree distribution
        degree_counts = defaultdict(int)
        
        for edge in network_data["edges"]:
            degree_counts[edge["agent_id"]] += 1
            degree_counts[edge["merchant_id"]] += 1
        
        if not degree_counts:
            return 0.0
        
        max_degree = max(degree_counts.values())
        avg_degree = sum(degree_counts.values()) / len(degree_counts)
        
        return (max_degree - avg_degree) / max_degree if max_degree > 0 else 0.0

    def _detect_communities(self, network_data: Dict[str, Any]) -> int:
        """Detect number of communities (simplified)"""
        
        # Simplified community detection based on connectivity
        if not network_data["edges"]:
            return len(network_data["nodes"])
        
        # Assume 1 community for connected networks
        return 1

    def _count_bridge_edges(self, network_data: Dict[str, Any]) -> int:
        """Count bridge edges in network"""
        
        # Simplified bridge detection
        # In practice, this would use proper bridge detection algorithms
        return len(network_data["edges"]) // 4  # Rough estimate

    def _get_account_data(self, account_id: str) -> Optional[Dict[str, Any]]:
        """Get account data by ID"""
        query = "SELECT * FROM mule_accounts WHERE account_id = ?"
        rows = self.db.execute_query(query, (account_id,))
        return rows[0] if rows else None

    def _calculate_account_centrality(self, account_id: str) -> float:
        """Calculate account centrality in transaction network"""
        
        graph = self._build_transaction_graph()
        
        # Simple degree centrality
        out_degree = len(graph.get(account_id, {}))
        in_degree = sum(1 for neighbors in graph.values() if account_id in neighbors)
        
        total_nodes = len(graph) + sum(len(neighbors) for neighbors in graph.values())
        
        if total_nodes == 0:
            return 0.0
        
        return (out_degree + in_degree) / total_nodes

    def _calculate_amount_anomaly_score(self, account_id: str) -> float:
        """Calculate amount anomaly score"""
        
        # Get average transaction amounts for this account
        query = """
        SELECT AVG(total_amount) as avg_amount
        FROM agent_networks 
        WHERE agent_id = ? OR merchant_id = ?
        """
        
        rows = self.db.execute_query(query, (account_id, account_id))
        if not rows or not rows[0]["avg_amount"]:
            return 0.0
        
        avg_amount = rows[0]["avg_amount"]
        
        # Flag unusually high amounts (assuming $10,000 is high)
        return min(1.0, avg_amount / 10000.0)

    def _calculate_temporal_pattern_score(self, transaction_patterns: Dict[str, Any]) -> float:
        """Calculate temporal pattern risk score"""
        
        if not transaction_patterns:
            return 0.0
        
        # Risk factors: short intervals, weekend activity
        avg_interval = transaction_patterns.get("avg_interval_hours", 24)
        weekend_activity = transaction_patterns.get("weekend_activity", False)
        
        interval_risk = 1.0 - min(1.0, avg_interval / 24.0)
        weekend_risk = 0.3 if weekend_activity else 0.0
        
        return min(1.0, interval_risk + weekend_risk)

    def _calculate_geographic_anomaly_score(self, account_id: str) -> float:
        """Calculate geographic anomaly score"""
        
        # Simplified - would need location data in practice
        return 0.2  # Default low risk

    def _update_mule_score(self, account_id: str, mule_score: float) -> None:
        """Update mule score in database"""
        
        query = """
        UPDATE mule_accounts 
        SET mule_score = ?, updated_at = ?
        WHERE account_id = ?
        """
        
        self.db.execute_update(query, (mule_score, datetime.now().isoformat(), account_id))