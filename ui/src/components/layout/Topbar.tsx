import { useTheme } from '@/theme/theme-provider';
import { Search, Bell, Sun, Moon } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useNavigate } from 'react-router-dom';

export function Topbar() {
    const { theme, setTheme } = useTheme();
    const navigate = useNavigate();

    return (
        <header className="h-16 px-6 border-b border-slate-200 dark:border-slate-800 bg-background/80 backdrop-blur-xl flex items-center justify-between sticky top-0 z-40">
            {/* Search */}
            <div className="w-96 relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 w-4 h-4" />
                <Input
                    placeholder="Search alerts, features..."
                    className="pl-9 bg-slate-100/50 dark:bg-slate-800/50 border-slate-200 dark:border-slate-700 focus:bg-background transition-all"
                />
            </div>

            {/* Actions */}
            <div className="flex items-center gap-4">
                <Button
                    variant="ghost"
                    size="icon"
                    className="relative text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800"
                    onClick={() => navigate('/app/alerts')}
                >
                    <Bell size={20} />
                    <span className="absolute top-2 right-2 w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                </Button>

                <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
                    className="text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800"
                >
                    <Sun className="h-[1.2rem] w-[1.2rem] rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
                    <Moon className="absolute h-[1.2rem] w-[1.2rem] rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
                    <span className="sr-only">Toggle theme</span>
                </Button>

                <div className="h-8 w-[1px] bg-slate-200 dark:bg-slate-700 mx-2" />

                <div className="flex items-center gap-3">
                    <div className="text-right hidden md:block">
                        <p className="text-sm font-semibold text-slate-900 dark:text-slate-100">Admin User</p>
                        <p className="text-xs text-slate-500 dark:text-slate-400">Super Admin</p>
                    </div>
                    <div className="h-9 w-9 bg-gradient-to-tr from-accent to-blue-600 rounded-full flex items-center justify-center text-white font-bold shadow-lg">
                        M
                    </div>
                </div>
            </div>
        </header>
    );
}
