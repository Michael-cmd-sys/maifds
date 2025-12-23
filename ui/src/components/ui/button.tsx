import * as React from 'react';
import { cn } from '@/utils/cn';
import { Loader2 } from 'lucide-react';



// Manually defining VariantProps since we didn't install class-variance-authority yet but implemented logic similar to it.
// Actually I need to install `class-variance-authority`. I missed in dependencies.
// I will just use manual cn logic to avoid extra dependency for now, or just install it.
// Installing it is better for cleaner code.
// I'll assume I have it or will add it. I'll add `class-variance-authority` to next install list.
// For now, I'll rewrite without cva to be safe or just use a simpler approach.

// Let's use a simpler approach without `cva` for now to avoid install delay, or just run install in parallel.
// Actually, `clsx` and `tailwind-merge` are enough.

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link' | 'glass' | 'glass-accent';
    size?: 'default' | 'sm' | 'lg' | 'icon';
    isLoading?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
    ({ className, variant = 'default', size = 'default', isLoading, children, ...props }, ref) => {

        const variants = {
            default: 'bg-primary text-white hover:bg-slate-800 shadow-lg shadow-primary/10',
            destructive: 'bg-red-500 text-white hover:bg-red-600 shadow-md hover:shadow-red-500/25',
            outline: 'border border-slate-200 dark:border-slate-700 bg-transparent hover:bg-slate-100 dark:hover:bg-slate-800',
            secondary: 'bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-slate-100 hover:bg-slate-200 dark:hover:bg-slate-700',
            ghost: 'hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-slate-100',
            link: 'text-primary underline-offset-4 hover:underline',
            glass: 'bg-white/40 dark:bg-slate-900/40 backdrop-blur-md border border-white/20 dark:border-white/10 hover:bg-white/50 dark:hover:bg-slate-900/60 text-slate-900 dark:text-white shadow-sm',
            'glass-accent': 'bg-accent/10 backdrop-blur-md border border-accent/20 hover:bg-accent/20 text-accent font-semibold shadow-[0_0_15px_rgba(34,211,238,0.1)] hover:shadow-[0_0_20px_rgba(34,211,238,0.3)]',
        };

        const sizes = {
            default: 'h-10 px-4 py-2',
            sm: 'h-9 rounded-md px-3 text-xs',
            lg: 'h-11 rounded-md px-8 text-base',
            icon: 'h-10 w-10 p-2 flex items-center justify-center',
        };

        return (
            <button
                className={cn(
                    'inline-flex items-center justify-center whitespace-nowrap rounded-lg font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 transition-all duration-300',
                    variants[variant],
                    sizes[size],
                    className
                )}
                ref={ref}
                disabled={isLoading || props.disabled}
                {...props}
            >
                {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                {children}
            </button>
        );
    }
);
Button.displayName = 'Button';

export { Button };
