import { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { apiClient } from '@/api/client';
import { ENDPOINTS } from '@/api/endpoints';
import { cn } from '@/utils/cn';
import {
    Play, Copy, Code,
    Terminal, ShieldCheck, Phone, MousePointer, Activity,
    Users, AlertTriangle, Smartphone, BarChart3, Heart
} from 'lucide-react';
import { CURL_TESTS } from '@/data/curl-tests';
import { useToast } from "@/hooks/use-toast";

const FEATURES = [
    {
        id: 'health_check',
        name: 'Health Check',
        path: '/v1/health',
        method: 'GET',
        icon: Heart,
        desc: 'Check API availability.',
        sample: {}
    },
    {
        id: 'blacklist_stats',
        name: 'Blacklist Stats',
        path: '/v1/blacklist/stats',
        method: 'GET',
        icon: BarChart3,
        desc: 'View statistics for blacklist entries.',
        sample: {}
    },
    {
        id: 'crs_stats',
        name: 'Reputation Stats',
        path: '/v1/customer-reputation/stats',
        method: 'GET',
        icon: BarChart3,
        desc: 'View statistics for customer reputation system.',
        sample: {}
    },
    {
        id: 'audit_stats',
        name: 'Audit Stats',
        path: '/v1/governance/audit/stats?limit=20',
        method: 'GET',
        icon: BarChart3,
        desc: 'View governance audit statistics.',
        sample: {}
    },
    {
        id: 'call_defense',
        name: 'Call-Triggered Defense',
        path: ENDPOINTS.FEATURES.CALL_DEFENSE,
        method: 'POST',
        icon: Phone,
        desc: 'Analyze call metadata for fraud patterns.',
        sample: {
            "call_id": "call_123456789",
            "caller_number": "+1234567890",
            "receiver_number": "+0987654321",
            "duration_seconds": 45,
            "audio_features": { "pitch": 120.5, "jitter": 0.02, "shimmer": 0.05 }
        }
    },
    {
        id: 'phishing',
        name: 'Phishing Detector',
        path: ENDPOINTS.FEATURES.PHISHING_SCORE,
        method: 'POST',
        icon: MousePointer,
        desc: 'Score URLs and ad content for phishing risk.',
        sample: {
            "url": "http://suspicious-bank-login.com",
            "content_snippet": "Click here to reset your password immediately.",
            "referrer": "social_media_ad"
        }
    },
    {
        id: 'click_tx',
        name: 'Click-to-Transaction',
        path: ENDPOINTS.FEATURES.CLICK_TX_CORRELATION,
        method: 'POST',
        icon: Activity,
        desc: 'Correlate user clicks with transaction intent.',
        sample: {
            "user_id": "u_998877",
            "session_id": "sess_555",
            "clicks": [{ "element": "login_btn", "timestamp": 1678886400 }, { "element": "transfer_menu", "timestamp": 1678886405 }]
        }
    },
    {
        id: 'reputation_report',
        name: 'Submit Reputation Report',
        path: ENDPOINTS.FEATURES.REPUTATION_SUBMIT,
        method: 'POST',
        icon: Users,
        desc: 'Report a fraudulent entity to the community.',
        sample: {
            "reporter_id": "u_trusted_01",
            "target_id": "u_suspect_99",
            "report_type": "scam_attempt",
            "description": "Attempted to impersonate bank staff via phone."
        }
    },
    {
        id: 'blacklist_add',
        name: 'Add to Blacklist',
        path: ENDPOINTS.BLACKLIST.ADD,
        method: 'POST',
        icon: ShieldCheck,
        desc: 'Add an entity to the global watchlist.',
        sample: {
            "value": "+1234567890",
            "list_type": "phone_number",
            "reason": "Confirmed scammer"
        }
    },
    {
        id: 'pre_tx',
        name: 'Pre-Transaction Warning',
        path: ENDPOINTS.FEATURES.PRE_TX_WARNING,
        method: 'POST',
        icon: AlertTriangle,
        desc: 'Analyze transaction context before execution.',
        sample: {
            "user_id": "u_123",
            "transaction_amount": 5000.00,
            "currency": "USD",
            "beneficiary_id": "acc_987654",
            "device_id": "dev_xyz_789"
        }
    },
    {
        id: 'telco_notify',
        name: 'Telco Notification',
        path: ENDPOINTS.FEATURES.TELCO_NOTIFY,
        method: 'POST',
        icon: Smartphone,
        desc: 'Trigger network-level alerts via Telco API.',
        sample: {
            "phone_number": "+15550001234",
            "alert_type": "sim_swap_risk",
            "severity": "high"
        }
    },
    {
        id: 'user_sms_alert',
        name: 'User SMS Alert',
        path: ENDPOINTS.FEATURES.SMS_ALERT,
        method: 'POST',
        icon: Smartphone,
        desc: 'Send direct SMS warning to user.',
        sample: {
            "user_id": "u_567",
            "phone": "+233555555555",
            "message": "fraud_warning_login"
        }
    },
    {
        id: 'orchestrate',
        name: 'AI Orchestrator',
        path: ENDPOINTS.FEATURES.ORCHESTRATOR,
        method: 'POST',
        icon: Activity,
        desc: 'Analyze context and decide on actions (SMS/Webhook).',
        sample: {
            "user_phone": "+233555555555",
            "incident_id": "inc_123",
            "suspected_number": "+233500000000",
            "call_triggered_defense": {
                "risk_score": 0.85,
                "action": "block"
            },
            "proactive_pre_tx_warning": {
                "user_risk_score": 0.9
            }
        }
    },
    {
        id: 'privacy_classify',
        name: 'Privacy Classification',
        path: ENDPOINTS.GOVERNANCE.PRIVACY_CLASSIFY,
        method: 'POST',
        icon: Terminal,
        desc: 'Classify text data for PII.',
        sample: {
            "text": "My email is john.doe@example.com and phone is +1-555-0199."
        }
    },
    {
        id: 'audit_event',
        name: 'Log Audit Event',
        path: ENDPOINTS.GOVERNANCE.AUDIT_SEND,
        method: 'POST',
        icon: Code,
        desc: 'Log a system event for compliance.',
        sample: {
            "event_type": "data_access",
            "actor_id": "user_123",
            "resource_id": "file_abc",
            "status": "success"
        }
    }
];

export default function Playground() {
    const [selectedFeature, setSelectedFeature] = useState(FEATURES[0]);
    const [payload, setPayload] = useState(JSON.stringify(FEATURES[0].sample, null, 2));
    const [response, setResponse] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [status, setStatus] = useState<number | null>(null);
    const [activeTab, setActiveTab] = useState<'interactive' | 'tests'>('interactive');
    const [testCurls, setTestCurls] = useState<string[]>([]);
    const [selectedTestFeature, setSelectedTestFeature] = useState<any>(null);
    const { toast } = useToast();

    const loadTestCurls = (feature: any) => {
        setSelectedTestFeature(feature);
        setTestCurls(CURL_TESTS[feature.id] || [`curl -X POST ... # No pre-defined tests for ${feature.name}`]);
    };

    const handleFeatureSelect = (feature: typeof FEATURES[0]) => {
        setSelectedFeature(feature);
        setPayload(JSON.stringify(feature.sample, null, 2));
        setResponse(null);
        setStatus(null);
    };

    const runRequest = async () => {
        setLoading(true);
        setResponse(null);
        setStatus(null);
        try {
            let res;
            if (selectedFeature.method === 'GET') {
                res = await apiClient.get(selectedFeature.path);
            } else {
                const data = JSON.parse(payload);
                res = await apiClient.post(selectedFeature.path, data);
            }

            setResponse(res.data);
            setStatus(res.status);
            toast({
                title: "Request Successful",
                description: `Received status ${res.status}`,
                variant: "success",
            });
        } catch (error: any) {
            console.error(error);
            setResponse(error.response?.data || error.message);
            setStatus(error.response?.status || 500);
            toast({
                title: "Request Failed",
                description: error.response?.data?.detail || error.message || "Unknown error occurred",
                variant: "destructive",
            });
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-full gap-6">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold bg-gradient-to-r from-accent to-purple-500 bg-clip-text text-transparent">API Playground</h1>
                    <p className="text-slate-500 dark:text-slate-400">Interactive testing lab for MindSpore fraud defense models.</p>
                </div>
            </div>

            {/* Tabs Header */}
            <div className="flex items-center gap-4 border-b border-slate-200 dark:border-slate-800 pb-0">
                <button
                    onClick={() => setActiveTab('interactive')}
                    className={cn(
                        "pb-3 text-sm font-medium transition-colors border-b-2",
                        activeTab === 'interactive'
                            ? "border-accent text-accent"
                            : "border-transparent text-slate-500 hover:text-slate-700 dark:hover:text-slate-300"
                    )}
                >
                    Interactive Lab
                </button>
                <button
                    onClick={() => setActiveTab('tests')}
                    className={cn(
                        "pb-3 text-sm font-medium transition-colors border-b-2",
                        activeTab === 'tests'
                            ? "border-accent text-accent"
                            : "border-transparent text-slate-500 hover:text-slate-700 dark:hover:text-slate-300"
                    )}
                >
                    Test Curls Library
                </button>
            </div>

            {activeTab === 'interactive' ? (
                <div className="flex flex-1 gap-6 overflow-hidden">
                    {/* Sidebar: Feature List */}
                    <Card className="w-1/4 glass-panel flex flex-col overflow-hidden border-0">
                        <div className="p-4 border-b border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/50">
                            <h3 className="font-semibold flex items-center gap-2">
                                <Terminal size={16} /> Available Endpoints
                            </h3>
                        </div>
                        <div className="flex-1 overflow-y-auto p-2 space-y-1">
                            {FEATURES.map(feature => (
                                <button
                                    key={feature.id}
                                    onClick={() => handleFeatureSelect(feature)}
                                    className={cn(
                                        "w-full text-left px-3 py-3 rounded-lg text-sm transition-all flex items-start gap-3 group",
                                        selectedFeature.id === feature.id
                                            ? "bg-slate-200 dark:bg-slate-800 shadow-sm ring-1 ring-slate-300 dark:ring-slate-700"
                                            : "hover:bg-slate-100 dark:hover:bg-slate-800/10 text-slate-600 dark:text-slate-400"
                                    )}
                                >
                                    <div className={cn(
                                        "p-2 rounded-md shrink-0 transition-colors",
                                        selectedFeature.id === feature.id ? "bg-white dark:bg-slate-900 text-accent" : "bg-slate-100 dark:bg-slate-800 text-slate-500"
                                    )}>
                                        <feature.icon size={16} />
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <div className="font-medium truncate text-slate-900 dark:text-slate-200">{feature.name}</div>
                                        <div className="text-[10px] uppercase font-bold text-slate-400 mt-0.5">{feature.method}</div>
                                    </div>
                                </button>
                            ))}
                        </div>
                    </Card>

                    {/* Main Content: Request/Response */}
                    <div className="flex-1 flex flex-col gap-4 overflow-hidden">
                        {/* Request URL Bar */}
                        <Card className="p-2 flex items-center gap-2 glass-card">
                            <Badge variant="outline" className={cn("px-3 py-1 font-mono uppercase",
                                selectedFeature.method === 'GET' ? "text-blue-500 border-blue-500" : "text-green-500 border-green-500"
                            )}>
                                {selectedFeature.method}
                            </Badge>
                            <code className="text-sm font-mono text-slate-600 dark:text-slate-300 flex-1 px-2">
                                {selectedFeature.path}
                            </code>
                            <Button
                                className="bg-slate-800 dark:bg-white text-white dark:text-slate-900 hover:bg-slate-900 dark:hover:bg-slate-200"
                                onClick={runRequest}
                                disabled={loading}
                            >
                                {loading ? <Activity className="animate-spin mr-2" /> : <Play className="mr-2" size={16} />}
                                Run Request
                            </Button>
                        </Card>

                        <div className="flex-1 grid grid-rows-2 gap-4 min-h-0">
                            {/* Request Body */}
                            <Card className="glass-card flex flex-col overflow-hidden">
                                <div className="p-3 border-b border-white/10 flex justify-between items-center bg-slate-50/50 dark:bg-slate-900/50">
                                    <span className="font-semibold text-sm">Request Body (JSON)</span>
                                    {selectedFeature.method === 'GET' && <span className="text-xs text-yellow-500 italic">GET requests ignore body</span>}
                                    {/* <Badge variant="outline" className="font-mono text-[10px]">application/json</Badge> */}
                                </div>
                                <div className="relative flex-1">
                                    {selectedFeature.method !== 'GET' && (
                                        <div className="absolute top-0 right-0 p-2 z-10 flex">
                                            <Button
                                                variant="outline"
                                                size="sm"
                                                className="h-8 text-xs bg-slate-200 dark:bg-slate-700 border-none text-slate-700 dark:text-slate-200 hover:bg-slate-300 dark:hover:bg-slate-600"
                                                onClick={() => setPayload(JSON.stringify(selectedFeature.sample, null, 2))}
                                            >
                                                Reset to Sample
                                            </Button>
                                        </div>
                                    )}
                                    <textarea
                                        value={payload}
                                        onChange={(e) => setPayload(e.target.value)}
                                        className={cn("w-full h-full font-mono text-sm p-4 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-slate-100 border border-slate-200 dark:border-slate-700 focus:ring-2 focus:ring-accent focus:outline-none resize-none",
                                            selectedFeature.method === 'GET' && "opacity-50 cursor-not-allowed"
                                        )}
                                        spellCheck={false}
                                        readOnly={selectedFeature.method === 'GET'}
                                    />
                                </div>
                            </Card>

                            {/* Response Section */}
                            <Card className="glass-card flex flex-col overflow-hidden">
                                <div className="p-3 border-b border-white/10 flex justify-between items-center bg-slate-50/50 dark:bg-slate-900/50">
                                    <span className="font-semibold text-sm flex items-center gap-2">
                                        Response
                                    </span>
                                    <span className="text-xs text-slate-500">
                                        {response ? `${JSON.stringify(response).length} bytes` : 'Waiting...'}
                                    </span>
                                </div>
                                <div className="relative flex-1">
                                    <div className="absolute top-0 right-0 p-2 z-10">
                                        <span className={cn("text-xs font-bold px-2 py-1 rounded", status && status >= 200 && status < 300 ? 'bg-green-500/20 text-green-500' : 'bg-red-500/20 text-red-500')}>
                                            {status || '---'}
                                        </span>
                                    </div>
                                    <div className="w-full h-full font-mono text-sm p-4 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-slate-100 border border-slate-200 dark:border-slate-700 overflow-auto">
                                        {loading && <div className="text-slate-500 animate-pulse">Processing request...</div>}
                                        {!loading && !response && <div className="opacity-50 flex flex-col items-center justify-center h-full gap-2">
                                            <Code size={32} />
                                            <span>Run a request to see the matching response</span>
                                        </div>}
                                        {!loading && response && (
                                            <pre>{JSON.stringify(response, null, 2)}</pre>
                                        )}
                                    </div>
                                </div>
                            </Card>
                        </div>
                    </div>
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
                    {FEATURES.map(feature => (
                        <div key={feature.id} onClick={() => loadTestCurls(feature)} className="glass-panel p-4 cursor-pointer hover:border-accent border border-transparent transition-all group">
                            <div className="mb-3 p-2 bg-slate-100 dark:bg-slate-800 w-fit rounded-lg group-hover:bg-accent/10 group-hover:text-accent transition-colors">
                                <feature.icon size={24} />
                            </div>
                            <h3 className="font-bold text-sm mb-1">{feature.name}</h3>
                            <p className="text-xs text-slate-500 line-clamp-2">{feature.desc}</p>
                        </div>
                    ))}

                    {selectedTestFeature && (
                        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4" onClick={() => setSelectedTestFeature(null)}>
                            <div className="bg-background w-full max-w-4xl max-h-[80vh] rounded-xl shadow-2xl flex flex-col overflow-hidden border border-slate-800" onClick={e => e.stopPropagation()}>
                                <div className="p-4 border-b border-slate-200 dark:border-slate-800 flex justify-between items-center">
                                    <h2 className="font-bold flex items-center gap-2"><Terminal size={18} /> {selectedTestFeature.name} - Test Curls</h2>
                                    <Button size="sm" variant="ghost" onClick={() => setSelectedTestFeature(null)}>Close</Button>
                                </div>
                                <div className="flex-1 overflow-auto p-6 space-y-6 bg-slate-50 dark:bg-slate-950">
                                    {testCurls.map((curl, i) => (
                                        <div key={i} className="space-y-2">
                                            <div className="flex justify-between items-center">
                                                <h4 className="text-sm font-semibold text-slate-500">Test Case {i + 1}</h4>
                                                <Button size="sm" variant="outline" className="h-7 text-xs" onClick={() => {
                                                    // Load into interactive mode
                                                    setSelectedFeature(selectedTestFeature);
                                                    // Try to extract JSON from the curl, otherwise just use default sample
                                                    try {
                                                        const jsonMatch = curl.match(/-d '(\{.*?\})'/s);
                                                        if (jsonMatch && jsonMatch[1]) {
                                                            setPayload(JSON.stringify(JSON.parse(jsonMatch[1]), null, 2));
                                                        } else {
                                                            setPayload(JSON.stringify(selectedTestFeature.sample, null, 2));
                                                        }
                                                    } catch (e) {
                                                        setPayload(JSON.stringify(selectedTestFeature.sample, null, 2));
                                                    }
                                                    setActiveTab('interactive');
                                                    setSelectedTestFeature(null);
                                                }}>Load to Runner</Button>
                                            </div>
                                            <div className="bg-slate-900 rounded-lg p-4 font-mono text-xs text-slate-300 relative group overflow-x-auto">
                                                <pre>{curl}</pre>
                                                <Button size="icon" variant="ghost" className="absolute top-2 right-2 text-slate-500 hover:text-white" onClick={() => navigator.clipboard.writeText(curl)}>
                                                    <Copy size={14} />
                                                </Button>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
