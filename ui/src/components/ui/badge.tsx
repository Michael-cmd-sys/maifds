import * as React from "react"
import { cn } from "@/utils/cn"

// Since I decided not to use cva earlier, I will convert this to basic object mapping or use cva if I installed it (I didn't actually check cva install, but I'll stick to manual mapping for consistency with Button if I want to be safe, but actually let's just stick to the pattern I established: manual mapping).

interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
    variant?: "default" | "secondary" | "destructive" | "outline" | "success" | "warning" | "glass";
}

function Badge({ className, variant = "default", ...props }: BadgeProps) {
    const variants = {
        default: "border-transparent bg-primary text-primary-foreground hover:bg-primary/80 text-white",
        secondary: "border-transparent bg-secondary text-secondary-foreground hover:bg-secondary/80 bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-slate-100",
        destructive: "border-transparent bg-red-500 text-white hover:bg-red-600",
        outline: "text-foreground border-slate-200 dark:border-slate-800",
        success: "border-transparent bg-green-500/15 text-green-700 dark:text-green-400 hover:bg-green-500/25 border-green-500/20",
        warning: "border-transparent bg-yellow-500/15 text-yellow-700 dark:text-yellow-400 hover:bg-yellow-500/25 border-yellow-500/20",
        glass: "glass text-foreground hover:bg-white/20",
    }

    return (
        <div
            className={cn(
                "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",
                variants[variant],
                className
            )}
            {...props}
        />
    )
}

export { Badge }
