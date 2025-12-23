import * as React from "react";
import { cn } from "@/utils/cn";

export interface DataTableProps extends React.HTMLAttributes<HTMLDivElement> {
    headers: string[];
    data: any[];
    renderRow: (item: any, index: number) => React.ReactNode;
    emptyMessage?: string;
}

export function DataTable({
    headers,
    data,
    renderRow,
    emptyMessage = "No data available",
    className,
    ...props
}: DataTableProps) {
    return (
        <div className={cn("glass-panel overflow-hidden", className)} {...props}>
            <div className="overflow-x-auto">
                <table className="w-full text-sm text-left">
                    <thead className="text-xs text-slate-500 dark:text-slate-400 uppercase bg-slate-100/50 dark:bg-slate-800/50 border-b border-slate-200 dark:border-slate-700">
                        <tr>
                            {headers.map((header, i) => (
                                <th key={i} className="px-6 py-3 font-medium tracking-wider">
                                    {header}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                        {data && data.length > 0 ? (
                            data.map((item, index) => renderRow(item, index))
                        ) : (
                            <tr>
                                <td
                                    colSpan={headers.length}
                                    className="px-6 py-12 text-center text-slate-500 dark:text-slate-400 italic"
                                >
                                    {emptyMessage}
                                </td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

export function TableRow({ children, className, ...props }: React.HTMLAttributes<HTMLTableRowElement>) {
    return (
        <tr className={cn("hover:bg-slate-50/50 dark:hover:bg-slate-800/50 transition-colors", className)} {...props}>
            {children}
        </tr>
    );
}

export function TableCell({ children, className, ...props }: React.TdHTMLAttributes<HTMLTableCellElement>) {
    return (
        <td className={cn("px-6 py-4 whitespace-nowrap", className)} {...props}>
            {children}
        </td>
    );
}
