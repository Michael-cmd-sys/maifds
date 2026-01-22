import { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { apiClient } from '@/api/client';
import { ENDPOINTS } from '@/api/endpoints';
import { ShieldAlert, ShieldCheck, Search, PlusCircle, Trash2, Database, RotateCw } from 'lucide-react';
import { motion } from 'framer-motion';
import { useToast } from "@/hooks/use-toast"

export default function Blacklist() {
    const { toast } = useToast()
    const [stats, setStats] = useState<any>(null);
    const [checkValue, setCheckValue] = useState('');
    const [checkResult, setCheckResult] = useState<any>(null);
    const [addValue, setAddValue] = useState('');
    const [addType, setAddType] = useState('phone_number');
    const [removeValue, setRemoveValue] = useState('');
    const [loading, setLoading] = useState<string | null>(null);

    const fetchStats = async () => {
        try {
            const res = await apiClient.get(ENDPOINTS.STATS.BLACKLIST);
            // Backend returns { feature: "...", result: { ... } }
            setStats(res.data.result || res.data);
        } catch (error) {
            console.error("Failed to fetch blacklist stats");
        }
    };

    useEffect(() => {
        fetchStats();
    }, []);

    const handleCheck = async () => {
        setLoading('check');
        try {
            // Basic heuristic to determine type for this simple UI
            const isPhone = /^[+\d]+$/.test(checkValue);
            const payload = isPhone ? { phone_number: checkValue } : { url: checkValue };

            const res = await apiClient.post(ENDPOINTS.FEATURES.BLACKLIST_CHECK, payload);
            setCheckResult(res.data.result || res.data);
        } catch (e) {
            setCheckResult({ error: "Check failed" });
        } finally {
            setLoading(null);
        }
    };

    const handleAdd = async () => {
        if (!addValue) return;
        setLoading('add');
        try {
            await apiClient.post(ENDPOINTS.BLACKLIST.ADD, {
                value: addValue,
                list_type: addType,
                reason: "Manual Admin Addition"
            });
            setAddValue('');
            await fetchStats();
            toast({
                title: "Entity Blacklisted",
                description: `${addValue} has been added to the blacklist.`,
                className: "bg-red-500 border-red-600 text-white",
            })
        } catch (e) {
            toast({
                title: "Operation Failed",
                description: "Failed to add to blacklist.",
                variant: "destructive",
            })
        } finally {
            setLoading(null);
        }
    };

    const handleRemove = async () => {
        if (!removeValue) return;
        setLoading('remove');
        try {
            await apiClient.post(ENDPOINTS.BLACKLIST.REMOVE, {
                value: removeValue,
                list_type: "phone_number" // Simplification for UI demo
            });
            setRemoveValue('');
            await fetchStats();
            toast({
                title: "Entity Removed",
                description: "Successfully removed from blacklist.",
            })
        } catch (e) {
            toast({
                title: "Removal Failed",
                description: "Failed to verify removal or entity not found.",
                variant: "destructive",
            })
        } finally {
            setLoading(null);
        }
    };

    return (
        <div className="space-y-8">
            <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-red-500 to-orange-500 bg-clip-text text-transparent flex items-center gap-3">
                    <Database className="text-red-500" /> Blacklist Management
                </h1>
                <p className="text-slate-500 dark:text-slate-400">Manage blocked entities and high-risk identifiers.</p>
            </div>

            {/* Stats Cards */}
            <div className="grid gap-6 md:grid-cols-4">
                <Card className="glass-card border-none bg-red-500/10 text-red-600">
                    <CardHeader className="pb-2"><CardTitle className="text-sm">Total Blocked</CardTitle></CardHeader>
                    <CardContent><div className="text-3xl font-bold">{stats?.database_entries || 0}</div></CardContent>
                </Card>
                <Card className="glass-card">
                    <CardHeader className="pb-2"><CardTitle className="text-sm">Phone Numbers</CardTitle></CardHeader>
                    <CardContent><div className="text-3xl font-bold">{stats?.bloom_filter_numbers?.num_elements || 0}</div></CardContent>
                </Card>
                <Card className="glass-card">
                    <CardHeader className="pb-2"><CardTitle className="text-sm">URLs / Domains</CardTitle></CardHeader>
                    <CardContent><div className="text-3xl font-bold">{stats?.bloom_filter_urls?.num_elements || 0}</div></CardContent>
                </Card>
                <Card className="glass-card">
                    <CardHeader className="pb-2"><CardTitle className="text-sm">Devices</CardTitle></CardHeader>
                    <CardContent><div className="text-3xl font-bold">{stats?.bloom_filter_devices?.num_elements || 0}</div></CardContent>
                </Card>
            </div>

            <div className="grid gap-8 md:grid-cols-2">

                {/* Check Status */}
                <Card className="glass-panel">
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2"><Search className="w-5 h-5 text-accent" /> Check Entity Status</CardTitle>
                        <CardDescription>Verify if a phone number or URL is currently blacklisted.</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="flex gap-2">
                            <Input
                                placeholder="Enter Phone (+123...) or URL"
                                value={checkValue}
                                onChange={(e) => setCheckValue(e.target.value)}
                            />
                            <Button onClick={handleCheck} disabled={!!loading} className="text-white dark:text-slate-900">
                                {loading === 'check' ? <RotateCw className="animate-spin" /> : "Check"}
                            </Button>
                        </div>

                        {checkResult && (
                            <motion.div
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className={`p-4 rounded-lg border flex items-start gap-3 ${checkResult.listed ? 'bg-red-500/10 border-red-500/20 text-red-600' : 'bg-green-500/10 border-green-500/20 text-green-600'}`}
                            >
                                {checkResult.listed ? <ShieldAlert className="mt-1" /> : <ShieldCheck className="mt-1" />}
                                <div>
                                    <h4 className="font-bold">{checkResult.listed ? 'BLACKLISTED' : 'CLEAN'}</h4>
                                    <p className="text-sm opacity-80">{checkResult.reason || "No records found for this entity."}</p>
                                    {checkResult.listed && <Badge variant="destructive" className="mt-2">Severity: {checkResult.severity || 'High'}</Badge>}
                                </div>
                            </motion.div>
                        )}
                    </CardContent>
                </Card>

                {/* Actions */}
                <div className="space-y-6">

                    {/* Add Form */}
                    <Card className="glass-panel border-l-4 border-l-red-500">
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2"><PlusCircle className="w-5 h-5" /> Add to Blacklist</CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="grid grid-cols-3 gap-2">
                                <div className="col-span-2">
                                    <Label>Value</Label>
                                    <Input value={addValue} onChange={(e) => setAddValue(e.target.value)} placeholder="+1234567890" />
                                </div>
                                <div>
                                    <Label>Type</Label>
                                    <select
                                        className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                                        value={addType}
                                        onChange={(e) => setAddType(e.target.value)}
                                    >
                                        <option value="phone_number">Phone</option>
                                        <option value="url">URL</option>
                                        <option value="device_id">Device</option>
                                    </select>
                                </div>
                            </div>
                            <Button onClick={handleAdd} disabled={!!loading} className="w-full" variant="destructive">
                                {loading === 'add' ? <RotateCw className="animate-spin mr-2" /> : <ShieldAlert className="mr-2 h-4 w-4" />}
                                Block Entity
                            </Button>
                        </CardContent>
                    </Card>

                    {/* Remove Form */}
                    <Card className="glass-panel opacity-80 hover:opacity-100">
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2 text-sm text-slate-500"><Trash2 className="w-4 h-4" /> Remove from Blacklist</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="flex gap-2">
                                <Input value={removeValue} onChange={(e) => setRemoveValue(e.target.value)} placeholder="Value to remove..." />
                                <Button variant="outline" onClick={handleRemove} disabled={!!loading}>Remove</Button>
                            </div>
                        </CardContent>
                    </Card>
                </div>

            </div>
        </div>
    );
}
