import { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { apiClient } from '@/api/client';
import { ENDPOINTS } from '@/api/endpoints';
import { Bell, AlertTriangle, ShieldAlert, Info, CheckCircle } from 'lucide-react';
import { cn } from '@/utils/cn';

export default function Alerts() {
    const [alerts, setAlerts] = useState<any[]>([]);
    const [filter, setFilter] = useState('all');

    useEffect(() => {
        // Determine which endpoint to use. 
        // /v1/customer-reputation/alerts is the main one listed for alerts
        apiClient.get(ENDPOINTS.FEATURES.ALERTS, { params: { threshold: 0.0 } }) // Get all
            .then(res => setAlerts(res.data.alerts || []))
            .catch(err => {
                console.error("Failed to fetch alerts", err);
                // Fallback mock data for demo if API is empty/unreachable
                setAlerts([
                    { id: 1, target_id: '123456789', reason: 'High frequency transactions detected', severity: 'high', created_at: new Date().toISOString() },
                    { id: 2, target_id: '987654321', reason: 'Reported by community as scammer', severity: 'critical', created_at: new Date(Date.now() - 3600000).toISOString() },
                    { id: 3, target_id: '555555555', reason: 'Suspicious URL access', severity: 'medium', created_at: new Date(Date.now() - 86400000).toISOString() },
                ]);
            });
    }, []);

    const getSeverityColor = (severity: string) => {
        switch (severity.toLowerCase()) {
            case 'critical': return 'bg-red-500 text-white border-red-600';
            case 'high': return 'bg-orange-500 text-white border-orange-600';
            case 'medium': return 'bg-yellow-500 text-white border-yellow-600';
            default: return 'bg-blue-500 text-white border-blue-600';
        }
    };

    const getSeverityIcon = (severity: string) => {
        switch (severity.toLowerCase()) {
            case 'critical': return <ShieldAlert className="h-5 w-5" />;
            case 'high': return <AlertTriangle className="h-5 w-5" />;
            case 'medium': return <Info className="h-5 w-5" />;
            default: return <Bell className="h-5 w-5" />;
        }
    };

    return (
        <div className="space-y-8">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <h1 className="text-3xl font-bold bg-gradient-to-r from-orange-400 to-red-500 bg-clip-text text-transparent flex items-center gap-3">
                        <Bell className="text-orange-400" /> System Alerts
                    </h1>
                    <p className="text-slate-500 dark:text-slate-400">Real-time notifications and threat warnings.</p>
                </div>
                <div className="flex items-center gap-2 bg-white dark:bg-slate-900 p-1 rounded-lg border border-slate-200 dark:border-slate-800">
                    <ButtonFilter label="All" active={filter === 'all'} onClick={() => setFilter('all')} />
                    <ButtonFilter label="Critical" active={filter === 'critical'} onClick={() => setFilter('critical')} />
                    <ButtonFilter label="High" active={filter === 'high'} onClick={() => setFilter('high')} />
                </div>
            </div>

            <div className="grid gap-4">
                {alerts.length === 0 && (
                    <div className="text-center py-20 text-slate-500">
                        <CheckCircle className="mx-auto h-12 w-12 text-green-500 mb-4 opacity-50" />
                        <h3 className="text-lg font-medium">All Clear</h3>
                        <p>No active threats detected in the system.</p>
                    </div>
                )}

                {alerts.filter((a: any) => filter === 'all' || a.severity === filter).map((alert: any, idx: number) => (
                    <Card key={idx} className="glass-panel hover:border-slate-300 dark:hover:border-slate-600 transition-all group">
                        <div className="flex items-center p-4 gap-4">
                            <div className={cn("p-3 rounded-full shrink-0 shadow-sm", getSeverityColor(alert.severity))}>
                                {getSeverityIcon(alert.severity)}
                            </div>

                            <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 mb-1">
                                    <h4 className="text-base font-bold truncate">Threat Detected: {alert.target_id}</h4>
                                    <Badge variant="outline" className="uppercase text-[10px] tracking-wider font-bold">
                                        {alert.severity}
                                    </Badge>
                                </div>
                                <p className="text-sm text-slate-600 dark:text-slate-300">
                                    {alert.reason}
                                </p>
                            </div>

                            <div className="text-right text-xs text-slate-400 whitespace-nowrap">
                                {new Date(alert.created_at).toLocaleString()}
                            </div>
                        </div>
                    </Card>
                ))}
            </div>
        </div>
    );
}

function ButtonFilter({ label, active, onClick }: { label: string, active: boolean, onClick: () => void }) {
    return (
        <button
            onClick={onClick}
            className={cn(
                "px-4 py-1.5 rounded-md text-sm font-medium transition-all",
                active
                    ? "bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-white shadow-sm"
                    : "text-slate-500 hover:text-slate-700 dark:hover:text-slate-300"
            )}
        >
            {label}
        </button>
    )
}
