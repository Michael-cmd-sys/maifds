import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { motion } from 'framer-motion';
import {
    ShieldCheck, Phone, MousePointer, Users, Search,
    AlertTriangle, Eye, Lock, Smartphone, ArrowRight, Activity
} from 'lucide-react';
import { Card } from '@/components/ui/card';

const FEATURES = [
    { icon: Phone, title: 'Call Defense', desc: 'Real-time detection of fraud calls using audio patterns.' },
    { icon: MousePointer, title: 'Phishing Detector', desc: 'Identify malicious ads and referral links instantly.' },
    { icon: Activity, title: 'Click-to-Tx', desc: 'Correlate user clicks with transaction anomalies.' },
    { icon: Users, title: 'Crowd Reputation', desc: 'Community-driven fraud reporting and scoring.' },
    { icon: ShieldCheck, title: 'Real-time Blacklist', desc: 'Instant blocking of known malicious entities.' },
    { icon: Search, title: 'Risk Profiling', desc: 'Deep dive analysis of agent and merchant behavior.' },
    { icon: Eye, title: 'Human-in-Loop', desc: 'Expert verification workflow for high-risk alerts.' },
    { icon: AlertTriangle, title: 'Pre-Tx Warning', desc: 'Proactive alerts before transaction completion.' },
    { icon: Smartphone, title: 'Telco Notification', desc: 'Automated warnings via carrier networks.' },
    { icon: Smartphone, title: 'SMS Alerts', desc: 'Direct-to-user SMS warnings for suspicious activity.' },
    { icon: Lock, title: 'Privacy Governance', desc: 'GDPR-compliant data handling and classification.' },
];

export default function Landing() {
    return (
        <div className="min-h-screen bg-background text-foreground flex flex-col relative overflow-hidden">
            {/* Background Decor */}
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[1000px] h-[600px] bg-primary/20 rounded-full blur-[120px] -z-10 pointer-events-none" />

            {/* Hero Section */}
            <section className="flex flex-col items-center justify-center pt-32 pb-20 px-6 text-center">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8 }}
                >
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-accent/10 text-accent text-sm font-medium mb-6 border border-accent/20">
                        <span className="relative flex h-2 w-2">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-accent"></span>
                        </span>
                        System Operational
                    </div>
                    <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-6 bg-gradient-to-r from-primary via-blue-600 to-accent bg-clip-text text-transparent pb-2">
                        MAIFDS
                    </h1>
                    <p className="text-xl md:text-2xl text-slate-600 dark:text-slate-300 max-w-3xl mx-auto mb-10 leading-relaxed">
                        MindSpore Powered Artificial Intelligence Fraud Defense System. <br />
                        <span className="text-base md:text-lg text-slate-500 font-normal">
                            Advanced protection for the digital financial ecosystem.
                        </span>
                    </p>

                    <div className="flex flex-wrap gap-4 justify-center">
                        <Link to="/app/overview">
                            <Button size="lg" className="rounded-full px-8 text-lg h-14 shadow-xl shadow-primary/20 text-white dark:text-white">
                                Launch Dashboard <ArrowRight className="ml-2 h-5 w-5" />
                            </Button>
                        </Link>
                        <a href="https://www.mindspore.cn/" target="_blank" rel="noopener noreferrer">
                            <Button size="lg" variant="outline" className="rounded-full px-8 text-lg h-14 bg-white/50 dark:bg-slate-900/50 backdrop-blur">
                                Learn MindSpore
                            </Button>
                        </a>
                    </div>
                </motion.div>
            </section>

            {/* Architecture Diagram */}
            <section className="py-16 px-6 bg-slate-50/50 dark:bg-slate-900/50 backdrop-blur-sm border-y border-slate-200 dark:border-slate-800">
                <div className="max-w-7xl mx-auto text-center">
                    <h2 className="text-3xl font-bold mb-12">System Architecture</h2>
                    <div className="flex flex-col md:flex-row items-center justify-center gap-8 md:gap-12 opacity-90">

                        {/* Client */}
                        <div className="p-6 rounded-2xl glass border border-slate-200 dark:border-slate-700 w-64">
                            <div className="h-12 w-12 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center mx-auto mb-4 text-blue-600 dark:text-blue-400">
                                <MousePointer />
                            </div>
                            <h3 className="font-bold text-lg mb-2">Frontend Client</h3>
                            <p className="text-sm text-slate-500">React + Vite UI</p>
                        </div>

                        <ArrowRight className="hidden md:block text-slate-400 rotate-90 md:rotate-0" />

                        {/* API */}
                        <div className="p-6 rounded-2xl glass border border-accent/30 shadow-[0_0_30px_rgba(34,211,238,0.1)] w-64 bg-accent/5 dark:bg-accent/5">
                            <div className="h-12 w-12 bg-accent/10 rounded-lg flex items-center justify-center mx-auto mb-4 text-accent">
                                <Activity />
                            </div>
                            <h3 className="font-bold text-lg mb-2">FastAPI Backend</h3>
                            <p className="text-sm text-slate-500">RESTful Services</p>
                        </div>

                        <ArrowRight className="hidden md:block text-slate-400 rotate-90 md:rotate-0" />

                        {/* AI Engine */}
                        <div className="p-6 rounded-2xl glass border border-purple-200 dark:border-purple-800 w-64">
                            <div className="h-12 w-12 bg-purple-100 dark:bg-purple-900/30 rounded-lg flex items-center justify-center mx-auto mb-4 text-purple-600 dark:text-purple-400">
                                <ShieldCheck />
                            </div>
                            <h3 className="font-bold text-lg mb-2">MindSpore Engine</h3>
                            <p className="text-sm text-slate-500">AI Inference</p>
                        </div>

                    </div>
                </div>
            </section>

            {/* Features Grid */}
            <section className="py-20 px-6 max-w-7xl mx-auto">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                    {FEATURES.map((feature, idx) => (
                        <motion.div
                            key={idx}
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            transition={{ delay: idx * 0.05 }}
                            viewport={{ once: true }}
                        >
                            <Card className="h-full hover:border-accent/40 group">
                                <div className="p-6 flex flex-col h-full">
                                    <div className="h-12 w-12 rounded-lg bg-slate-100 dark:bg-slate-800 flex items-center justify-center mb-4 group-hover:bg-accent/10 group-hover:text-accent transition-colors">
                                        <feature.icon className="h-6 w-6" />
                                    </div>
                                    <h3 className="font-bold text-lg mb-2">{feature.title}</h3>
                                    <p className="text-sm text-slate-500 dark:text-slate-400 leading-relaxed">
                                        {feature.desc}
                                    </p>
                                </div>
                            </Card>
                        </motion.div>
                    ))}
                </div>
            </section>

            {/* Footer */}
            <footer className="mt-auto py-8 border-t border-slate-200 dark:border-slate-800 text-center text-sm text-slate-500">
                <p className="mb-2">Huawei Innovation Competition 2025</p>
                <p className="font-semibold text-primary dark:text-accent flex items-center justify-center gap-2">
                    Powered by Huawei MindSpore <ShieldCheck size={14} />
                </p>
            </footer>
        </div>
    );
}
