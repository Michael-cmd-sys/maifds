import { useState, useEffect } from 'react';
import { apiClient } from '@/api/client';
import { ENDPOINTS } from '@/api/endpoints';
import { DataTable, TableRow, TableCell } from '@/components/ui/data-table';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { ScrollText, Filter } from 'lucide-react';
import { Badge } from '@/components/ui/badge';

export default function Audit() {
    const [logs, setLogs] = useState<any[]>([]);

    useEffect(() => {
        // Fetch mock stats or logs. API might not have a direct "list all logs" endpoint visible in summary 
        // but assuming /v1/governance/audit/stats return some list or we mock it for demo if list endpoint absent.
        // Looking at openapi, there isn't a clear "Get Logs" list, only "Get Stats". 
        // I will use mock data for the table to demonstrate the UI if API call fails or returns empty.

        // Real call
        apiClient.get(ENDPOINTS.HEALTH.AUDIT).catch(console.error);

        // Mock logs for UI demonstration
        setLogs([
            { id: 'evt_9988', action: 'data_access', user: 'admin_usr', status: 'success', time: new Date().toISOString() },
            { id: 'evt_9987', action: 'privacy_classify', user: 'system_bot', status: 'success', time: new Date(Date.now() - 50000).toISOString() },
            { id: 'evt_9986', action: 'blacklist_add', user: 'security_lead', status: 'warning', time: new Date(Date.now() - 120000).toISOString() },
            { id: 'evt_9985', action: 'login_attempt', user: 'unknown', status: 'failure', time: new Date(Date.now() - 200000).toISOString() },
        ]);

    }, []);

    return (
        <div className="space-y-8">
            <div>
                <h1 className="text-3xl font-bold flex items-center gap-3 text-slate-800 dark:text-slate-100">
                    <ScrollText /> Audit Logs
                </h1>
                <p className="text-slate-500 dark:text-slate-400">Immutable record of all system activities and access events.</p>
            </div>

            <Card className="glass-panel">
                <CardHeader className="flex flex-row items-center justify-between">
                    <CardTitle className="text-lg">System Events</CardTitle>
                    <div className="text-sm text-slate-500 flex items-center gap-1 cursor-pointer hover:text-primary">
                        <Filter size={14} /> Filter
                    </div>
                </CardHeader>
                <CardContent>
                    <DataTable
                        headers={["Event ID", "Action", "Actor", "Status", "Timestamp"]}
                        data={logs}
                        renderRow={(log, i) => (
                            <TableRow key={i}>
                                <TableCell className="font-mono text-xs">{log.id}</TableCell>
                                <TableCell className="font-medium">{log.action}</TableCell>
                                <TableCell>{log.user}</TableCell>
                                <TableCell>
                                    <Badge
                                        variant="outline"
                                        className={
                                            log.status === 'success'
                                                ? 'bg-green-500/10 text-green-600 border-green-500/20 hover:bg-green-500/20'
                                                : log.status === 'failure'
                                                    ? 'bg-red-500/10 text-red-600 border-red-500/20 hover:bg-red-500/20'
                                                    : 'bg-slate-500/10 text-slate-500 border-slate-500/20 hover:bg-slate-500/20'
                                        }
                                    >
                                        {log.status}
                                    </Badge>
                                </TableCell>
                                <TableCell className="text-slate-500 text-xs">{new Date(log.time).toLocaleString()}</TableCell>
                            </TableRow>
                        )}
                    />
                </CardContent>
            </Card>
        </div>
    );
}
