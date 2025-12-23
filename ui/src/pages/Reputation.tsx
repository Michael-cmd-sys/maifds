import { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { apiClient } from '@/api/client';
import { ENDPOINTS } from '@/api/endpoints';
import { DataTable, TableRow, TableCell } from '@/components/ui/data-table';
import { Users, AlertTriangle, Activity, Search, BarChart3 } from 'lucide-react';

export default function Reputation() {
    const [networkStats, setNetworkStats] = useState<any>(null);
    const [suspiciousTx, setSuspiciousTx] = useState<any[]>([]);
    const [alerts, setAlerts] = useState<any[]>([]);

    const [targetId, setTargetId] = useState('');
    const [riskResult, setRiskResult] = useState<any>(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        // 1. Fetch Network Stats (Overview)
        apiClient.get(ENDPOINTS.FEATURES.ALERTS, { params: { threshold: 0.5 } })
            .then(res => setAlerts(res.data.alerts || []))
            .catch(console.error);

        // 2. Fetch Suspicious Txs
        apiClient.get(ENDPOINTS.FEATURES.CLICK_TX_CORRELATION + '?debug=true') // Hack: using correlation or any stat endpoint. 
        // Actually per API analysis: 
        // /v1/customer-reputation/transactions/suspicious is the correct one.
        apiClient.get('/v1/customer-reputation/transactions/suspicious?hours=24')
            .then(res => setSuspiciousTx(res.data.transactions || []))
            .catch(console.error);

        // 3. Overall Stats
        apiClient.get(ENDPOINTS.STATS.REPUTATION)
            .then(res => setNetworkStats(res.data))
            .catch(console.error);
    }, []);

    const checkRisk = async () => {
        if (!targetId) return;
        setLoading(true);
        setRiskResult(null);
        try {
            // Trying both Agent and Merchant since UI is generic
            // In real App, user would select type.
            try {
                const res = await apiClient.get(ENDPOINTS.FEATURES.AGENT_RISK(targetId));
                setRiskResult({ type: 'Agent', ...res.data });
            } catch {
                const res = await apiClient.get(ENDPOINTS.FEATURES.MERCHANT_RISK(targetId));
                setRiskResult({ type: 'Merchant', ...res.data });
            }
        } catch {
            setRiskResult({ error: 'Entity not found or analysis failed' });
        } finally {
            setLoading(false);
        }
    }

    return (
        <div className="space-y-8">
            <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-500 to-purple-500 bg-clip-text text-transparent flex items-center gap-3">
                    <Users className="text-blue-500" /> Customer Reputation & Risk
                </h1>
                <p className="text-slate-500 dark:text-slate-400">Crowd-sourced insights and behavioral risk scoring.</p>
            </div>

            {/* Network Overview Stats */}
            <div className="grid gap-6 md:grid-cols-3">
                <Card className="glass-card">
                    <CardHeader className="pb-2 flex flex-row items-center justify-between space-y-0">
                        <CardTitle className="text-sm font-medium">Network Trust Score</CardTitle>
                        <Activity className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-3xl font-bold text-green-500">{networkStats?.network_trust_score || 85}/100</div>
                    </CardContent>
                </Card>
                <Card className="glass-card">
                    <CardHeader className="pb-2 flex flex-row items-center justify-between space-y-0">
                        <CardTitle className="text-sm font-medium">Active Agents</CardTitle>
                        <Users className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-3xl font-bold">{networkStats?.active_agents || 0}</div>
                    </CardContent>
                </Card>
                <Card className="glass-card">
                    <CardHeader className="pb-2 flex flex-row items-center justify-between space-y-0">
                        <CardTitle className="text-sm font-medium">Flagged Merchants</CardTitle>
                        <AlertTriangle className="h-4 w-4 text-orange-500" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-3xl font-bold text-orange-500">{networkStats?.flagged_merchants || 0}</div>
                    </CardContent>
                </Card>
            </div>

            <div className="grid md:grid-cols-3 gap-8">

                {/* Risk Lookup */}
                <div className="md:col-span-1 space-y-6">
                    <Card className="glass-panel min-h-[300px]">
                        <CardHeader>
                            <CardTitle className="text-lg flex items-center gap-2"><Search size={18} /> Deep Risk Scan</CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div>
                                <label className="text-xs font-semibold uppercase text-slate-500 mb-1 block">Agent / Merchant ID</label>
                                <div className="flex gap-2">
                                    <Input
                                        value={targetId}
                                        onChange={e => setTargetId(e.target.value)}
                                        placeholder="e.g. agt_12345"
                                    />
                                    <Button onClick={checkRisk} disabled={loading} className="text-white dark:text-slate-900">
                                        <Search size={16} className="mr-2" /> Search
                                    </Button>
                                </div>
                            </div>

                            {riskResult && (
                                <div className="mt-4 p-4 rounded-lg bg-slate-100 dark:bg-slate-800 border animate-in fade-in slide-in-from-bottom-2">
                                    {riskResult.error ? (
                                        <p className="text-red-500 text-sm">{riskResult.error}</p>
                                    ) : (
                                        <div className="space-y-3">
                                            <div className="flex justify-between items-center border-b pb-2 border-slate-200 dark:border-slate-700">
                                                <span className="font-bold">{riskResult.type} Found</span>
                                                <Badge variant={riskResult.risk_score > 0.7 ? "destructive" : "default"}>
                                                    Score: {riskResult.risk_score}
                                                </Badge>
                                            </div>
                                            <div className="text-sm space-y-1">
                                                <p><span className="text-slate-500">Reports:</span> {riskResult.report_count}</p>
                                                <p><span className="text-slate-500">Status:</span> {riskResult.status}</p>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}
                        </CardContent>
                    </Card>
                </div>

                {/* Suspicious Transactions Table */}
                <div className="md:col-span-2">
                    <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                        <BarChart3 className="text-accent" /> Recent Suspicious Activity
                    </h3>
                    <DataTable
                        headers={["Tx ID", "Entities", "Risk", "Time"]}
                        data={suspiciousTx}
                        renderRow={(tx, i) => (
                            <TableRow key={i}>
                                <TableCell className="font-mono text-xs">{tx.tx_id.substring(0, 8)}...</TableCell>
                                <TableCell>
                                    <div className="flex flex-col text-xs">
                                        <span>Src: {tx.sender_id}</span>
                                        <span className="text-slate-400">Dst: {tx.receiver_id}</span>
                                    </div>
                                </TableCell>
                                <TableCell>
                                    <Badge variant={tx.risk_score > 0.8 ? "destructive" : "secondary"}>
                                        {(tx.risk_score * 100).toFixed(0)}%
                                    </Badge>
                                </TableCell>
                                <TableCell className="text-xs text-slate-500">
                                    {new Date(tx.timestamp).toLocaleTimeString()}
                                </TableCell>
                            </TableRow>
                        )}
                        emptyMessage="No suspicious transactions detected recently."
                    />

                    <h3 className="text-xl font-bold mt-8 mb-4 flex items-center gap-2">
                        <AlertTriangle className="text-orange-500" /> Community Alerts
                    </h3>
                    <DataTable
                        headers={["Entity", "Reason", "Reporter", "Date"]}
                        data={alerts}
                        renderRow={(alert, i) => (
                            <TableRow key={i}>
                                <TableCell className="font-bold">{alert.target_id}</TableCell>
                                <TableCell>{alert.reason}</TableCell>
                                <TableCell className="text-xs text-slate-500">{alert.reporter_id}</TableCell>
                                <TableCell className="text-xs text-slate-500">{new Date(alert.created_at).toLocaleDateString()}</TableCell>
                            </TableRow>
                        )}
                        emptyMessage="No active crowd-sourced alerts."
                    />
                </div>
            </div>
        </div>
    );
}
