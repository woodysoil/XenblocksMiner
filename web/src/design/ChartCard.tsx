import { type ReactNode } from 'react';
import { tw } from './tokens';

interface ChartCardProps {
  title: string;
  subtitle?: string;
  action?: ReactNode;
  children: ReactNode;
}

export default function ChartCard({ title, subtitle, action, children }: ChartCardProps) {
  return (
    <div className={tw.card}>
      <div className="flex justify-between items-start px-5 pt-5 pb-3">
        <div className="min-w-0">
          <h3 className={`text-sm font-semibold ${tw.textPrimary}`}>{title}</h3>
          {subtitle && (
            <p className={`text-xs ${tw.textTertiary} mt-0.5`}>{subtitle}</p>
          )}
        </div>
        {action && <div className="ml-4 shrink-0">{action}</div>}
      </div>
      <div className="px-3 pb-4">{children}</div>
    </div>
  );
}
