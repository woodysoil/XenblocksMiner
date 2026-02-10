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
      <div className="flex justify-between items-center px-5 pt-4 pb-2">
        <div>
          <h3 className={`text-sm font-semibold ${tw.textPrimary}`}>{title}</h3>
          {subtitle && (
            <p className={`text-xs ${tw.textTertiary} mt-0.5`}>{subtitle}</p>
          )}
        </div>
        {action && <div>{action}</div>}
      </div>
      <div className="px-2 pb-4">{children}</div>
    </div>
  );
}
