import { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { cn } from '@/utils/cn';
import {
    LayoutDashboard,
    FlaskConical,
    ShieldAlert,
    Users,
    Scale,
    FileText,
    Bell,
    ChevronLeft,
    ChevronRight
} from 'lucide-react';
import { Button } from '@/components/ui/button';

const NAV_ITEMS = [
    { label: 'Overview', path: '/app/overview', icon: LayoutDashboard },
    { label: 'Playground', path: '/app/playground', icon: FlaskConical },
    { label: 'Blacklist', path: '/app/blacklist', icon: ShieldAlert },
    { label: 'Reputation', path: '/app/reputation', icon: Users },
    { label: 'Privacy', path: '/app/governance/privacy', icon: Scale },
    { label: 'Audit', path: '/app/governance/audit', icon: FileText },
    { label: 'Alerts', path: '/app/alerts', icon: Bell },
];

export function Sidebar() {
    const [collapsed, setCollapsed] = useState(false);

    return (
        <aside
            className={cn(
                "h-screen sticky top-0 bg-background/80 backdrop-blur-xl border-r border-slate-200 dark:border-slate-800 transition-all duration-300 flex flex-col z-50",
                collapsed ? "w-20" : "w-64"
            )}
        >
            {/* Header / Logo */}
            <div className="h-16 flex items-center justify-between px-4 border-b border-slate-200 dark:border-slate-800">
                {!collapsed && (
                    <>
                        <img
                            src="/logo/logo_light.png"
                            alt="MAIFDS"
                            className="h-8 w-auto dark:hidden"
                        />
                        <img
                            src="/logo/logo_dark.png"
                            alt="MAIFDS"
                            className="h-8 w-auto hidden dark:block"
                        />
                    </>
                )}
                <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => setCollapsed(!collapsed)}
                    className="ml-auto"
                >
                    {collapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
                </Button>
            </div>

            {/* Nav Items */}
            <nav className="flex-1 py-6 px-3 space-y-1">
                {NAV_ITEMS.map((item) => (
                    <NavLink
                        key={item.path}
                        to={item.path}
                        className={({ isActive }) =>
                            cn(
                                "flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200 group",
                                isActive
                                    ? "bg-accent/10 dark:bg-accent/20 text-accent dark:text-accent shadow-sm"
                                    : "text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-slate-200"
                            )
                        }
                    >
                        <item.icon size={20} className="shrink-0" />
                        {!collapsed && <span className="font-medium">{item.label}</span>}
                        {collapsed && (
                            <div className="absolute left-full ml-2 px-2 py-1 bg-slate-800 text-white text-xs rounded opacity-0 group-hover:opacity-100 pointer-events-none whitespace-nowrap z-50">
                                {item.label}
                            </div>
                        )}
                    </NavLink>
                ))}
            </nav>

            {/* Footer */}
            <div className="p-4 border-t border-slate-200 dark:border-slate-800">
                {!collapsed ? (
                    <div className="bg-slate-100 dark:bg-[#1A1A1D] rounded-lg p-3 text-center border border-slate-200 dark:border-slate-800">
                        <p className="text-xs font-semibold text-slate-500 dark:text-slate-400">Powered by</p>
                        <p className="text-sm font-bold text-slate-900 dark:text-white mt-0.5">Huawei MindSpore</p>
                    </div>
                ) : (
                    <div className="flex justify-center" title="Powered by Huawei MindSpore">
                        <div className="h-2 w-2 rounded-full bg-accent animate-pulse" />
                    </div>
                )}
            </div>
        </aside>
    );
}
