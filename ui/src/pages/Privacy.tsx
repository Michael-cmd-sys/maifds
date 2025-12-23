import { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { apiClient } from '@/api/client';
import { ENDPOINTS } from '@/api/endpoints';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { Lock, FileKey, Shield, Fingerprint } from 'lucide-react';
import { Badge } from '@/components/ui/badge';

const COLORS = ['#22D3EE', '#A78BFA', '#F472B6', '#FBBF24', '#34D399'];

export default function Privacy() {
    const [piiTypes, setPiiTypes] = useState<any>(null);
    const [mockChartData, setMockChartData] = useState<any[]>([]);

    useEffect(() => {
        // Fetch real stats
        apiClient.get(ENDPOINTS.STATS.PRIVACY)
            .then(() => {
                // Transform for chart if real data has labels
                // For now using mock data structure that matches typical response
                const chartData = [
                    { name: 'Phone', value: 400 },
                    { name: 'Email', value: 300 },
                    { name: 'Names', value: 300 },
                    { name: 'Loc', value: 200 },
                ];
                setMockChartData(chartData);
            })
            .catch(console.error);

        apiClient.get(ENDPOINTS.GOVERNANCE.PRIVACY_PII_TYPES)
            .then(res => setPiiTypes(res.data.pii_types || []))
            .catch(console.error);
    }, []);

    return (
        <div className="space-y-8">
            <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-emerald-400 to-cyan-500 bg-clip-text text-transparent flex items-center gap-3">
                    <Lock className="text-emerald-400" /> Privacy Governance
                </h1>
                <p className="text-slate-500 dark:text-slate-400">GDPR-compliant PII classification and protection metrics.</p>
            </div>

            <div className="grid gap-6 md:grid-cols-4">
                <Card className="glass-card bg-emerald-500/5 border-emerald-500/20">
                    <CardHeader className="pb-2"><CardTitle className="text-sm font-medium">PII Records Protected</CardTitle></CardHeader>
                    <CardContent><div className="text-3xl font-bold text-emerald-500">24.5k</div></CardContent>
                </Card>
                <Card className="glass-card">
                    <CardHeader className="pb-2"><CardTitle className="text-sm font-medium">Avg Anonymization Time</CardTitle></CardHeader>
                    <CardContent><div className="text-3xl font-bold">12ms</div></CardContent>
                </Card>
                <Card className="glass-card">
                    <CardHeader className="pb-2"><CardTitle className="text-sm font-medium">User Consent Rate</CardTitle></CardHeader>
                    <CardContent><div className="text-3xl font-bold">98.2%</div></CardContent>
                </Card>
                <Card className="glass-card">
                    <CardHeader className="pb-2"><CardTitle className="text-sm font-medium">Active Policies</CardTitle></CardHeader>
                    <CardContent><div className="text-3xl font-bold">14</div></CardContent>
                </Card>
            </div>

            <div className="grid md:grid-cols-2 gap-8">

                {/* Classification Chart */}
                <Card className="glass-panel h-[400px] flex flex-col">
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2"><Fingerprint className="text-accent" /> Sensitive Data Distribution</CardTitle>
                    </CardHeader>
                    <CardContent className="flex-1 min-h-0">
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie
                                    data={mockChartData}
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={60}
                                    outerRadius={100}
                                    fill="#8884d8"
                                    paddingAngle={5}
                                    dataKey="value"
                                >
                                    {mockChartData.map((_, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Pie>
                                <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }} />
                                <Legend />
                            </PieChart>
                        </ResponsiveContainer>
                    </CardContent>
                </Card>

                {/* PII Types & Methods */}
                <div className="space-y-6">
                    <Card className="glass-panel">
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2 text-base"><Shield className="w-4 h-4" /> Detected PII Types</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="flex flex-wrap gap-2">
                                {piiTypes && piiTypes.length > 0 ? piiTypes.map((type: string) => (
                                    <Badge key={type} variant="secondary" className="px-3 py-1 bg-slate-100 dark:bg-slate-800 border-none">
                                        {type}
                                    </Badge>
                                )) : (
                                    <p className="text-sm text-slate-500">Loading or system idle...</p>
                                )}
                            </div>
                        </CardContent>
                    </Card>

                    <Card className="glass-panel">
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2 text-base"><FileKey className="w-4 h-4" /> Active Protection Methods</CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="flex justify-between items-center p-3 rounded-lg bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
                                <div>
                                    <h4 className="font-bold text-sm">Differential Privacy (Noise)</h4>
                                    <p className="text-xs text-slate-500">Applied to aggregates</p>
                                </div>
                                <Badge className="bg-green-500/20 text-green-600 hover:bg-green-500/20">Active</Badge>
                            </div>
                            <div className="flex justify-between items-center p-3 rounded-lg bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
                                <div>
                                    <h4 className="font-bold text-sm">Hashing (SHA-256)</h4>
                                    <p className="text-xs text-slate-500">Identifiers & Keys</p>
                                </div>
                                <Badge className="bg-green-500/20 text-green-600 hover:bg-green-500/20">Active</Badge>
                            </div>
                            <div className="flex justify-between items-center p-3 rounded-lg bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
                                <div>
                                    <h4 className="font-bold text-sm">Redaction / Masking</h4>
                                    <p className="text-xs text-slate-500">Display output</p>
                                </div>
                                <Badge className="bg-green-500/20 text-green-600 hover:bg-green-500/20">Active</Badge>
                            </div>
                        </CardContent>
                    </Card>
                </div>

            </div>
        </div>
    );
}
