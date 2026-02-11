import { type ReactNode } from 'react';
import { tw, colors } from './tokens';
import Skeleton from './Skeleton';
import StatDelta from './StatDelta';

interface MetricCardProps {
  label: string;
  value: string | number;
  delta?: string | number;
  icon?: ReactNode;
  variant?: 'default' | 'accent' | 'success' | 'danger' | 'info' | 'warning';
  loading?: boolean;
}

const variantColor: Record<string, string> = {
  accent: colors.accent.DEFAULT,
  success: colors.success.DEFAULT,
  danger: colors.danger.DEFAULT,
  info: colors.info.DEFAULT,
  warning: colors.warning.DEFAULT,
};

const variantGlow: Record<string, string> = {
  accent: `0 -1px 12px ${colors.accent.glow}`,
  success: '0 -1px 12px rgba(14,203,129,0.15)',
  danger: '0 -1px 12px rgba(246,70,93,0.15)',
  info: '0 -1px 12px rgba(59,130,246,0.15)',
  warning: '0 -1px 12px rgba(240,185,11,0.15)',
};

export default function MetricCard({ label, value, delta, icon, variant = 'default', loading }: MetricCardProps) {
  const isPositive = typeof delta === 'string' && delta.startsWith('+');
  const isNegative = typeof delta === 'string' && delta.startsWith('-');

  return (
    <div
      className={`${tw.card} ${tw.cardHover} p-5 ${variantColor[variant] ? 'border-t-2' : ''}`}
      style={{
        ...(variantGlow[variant] ? { boxShadow: variantGlow[variant] } : {}),
        ...(variantColor[variant] ? { borderTopColor: variantColor[variant] } : {}),
      }}
    >
      <div className="flex items-center justify-between">
        <span className={`text-xs ${tw.textTertiary} uppercase tracking-wider`}>
          {label}
        </span>
        {icon && <span className={tw.textTertiary}>{icon}</span>}
      </div>

      {loading ? (
        <div className="mt-2 space-y-2">
          <Skeleton className="h-7 w-24" />
          {delta !== undefined && <Skeleton className="h-3 w-16" />}
        </div>
      ) : (
        <>
          <div className={`text-2xl font-bold ${tw.textPrimary} mt-1`} style={{ fontVariantNumeric: 'tabular-nums' }}>
            {value}
          </div>
          {delta !== undefined && (
            typeof delta === 'number' ? (
              <div className="mt-1">
                <StatDelta value={delta} />
              </div>
            ) : (
              <span
                className={`text-xs mt-1 inline-block ${
                  !isPositive && !isNegative ? tw.textSecondary : ''
                }`}
                style={isPositive ? { color: colors.success.DEFAULT } : isNegative ? { color: colors.danger.DEFAULT } : undefined}
              >
                {delta}
              </span>
            )
          )}
        </>
      )}
    </div>
  );
}
