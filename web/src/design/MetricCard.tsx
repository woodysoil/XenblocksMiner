import { type ReactNode } from 'react';
import { tw, colors } from './tokens';

interface MetricCardProps {
  label: string;
  value: string | number;
  delta?: string;
  icon?: ReactNode;
  variant?: 'default' | 'accent' | 'success' | 'danger';
}

const variantBorder: Record<string, string> = {
  accent: `border-t-2 border-t-[${colors.accent.DEFAULT}]`,
  success: `border-t-2 border-t-[${colors.success.DEFAULT}]`,
  danger: `border-t-2 border-t-[${colors.danger.DEFAULT}]`,
};

export default function MetricCard({ label, value, delta, icon, variant = 'default' }: MetricCardProps) {
  const isPositive = delta?.startsWith('+');
  const isNegative = delta?.startsWith('-');

  return (
    <div
      className={`${tw.card} ${tw.cardHover} p-5 ${variantBorder[variant] ?? ''}`}
    >
      <div className="flex items-center justify-between">
        <span className={`text-xs ${tw.textTertiary} uppercase tracking-wider`}>
          {label}
        </span>
        {icon && <span className={tw.textTertiary}>{icon}</span>}
      </div>
      <div className={`text-2xl font-bold ${tw.textPrimary} mt-1`}>{value}</div>
      {delta && (
        <span
          className={`text-xs mt-1 inline-block ${
            isPositive ? `text-[${colors.success.DEFAULT}]` : isNegative ? `text-[${colors.danger.DEFAULT}]` : tw.textSecondary
          }`}
        >
          {delta}
        </span>
      )}
    </div>
  );
}
