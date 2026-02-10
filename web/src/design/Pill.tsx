import { type ReactNode } from 'react';
import { tw, colors } from './tokens';

interface PillProps {
  icon?: ReactNode;
  label: string;
  value: string | number;
  color?: 'default' | 'accent' | 'success' | 'danger' | 'warning';
}

const valueColor: Record<string, string> = {
  default: colors.text.primary,
  accent:  colors.accent.DEFAULT,
  success: colors.success.DEFAULT,
  danger:  colors.danger.DEFAULT,
  warning: colors.warning.DEFAULT,
};

export default function Pill({ icon, label, value, color = 'default' }: PillProps) {
  return (
    <span className={`${tw.card} inline-flex items-center gap-2 rounded-full px-4 py-2`}>
      {icon && <span className={tw.textTertiary}>{icon}</span>}
      <span className={`text-xs ${tw.textTertiary}`}>{label}</span>
      <span className="font-semibold text-sm" style={{ color: valueColor[color] }}>
        {value}
      </span>
    </span>
  );
}
