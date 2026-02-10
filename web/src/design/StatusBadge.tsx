import { colors } from './tokens';

type Status = 'online' | 'offline' | 'mining' | 'leased' | 'available' | 'idle' | 'error';

interface StatusBadgeProps {
  status: Status;
  size?: 'sm' | 'md';
  label?: string;
}

const cfg: Record<Status, { color: string; label: string; pulse?: boolean }> = {
  online:    { color: colors.success.DEFAULT, label: 'Online' },
  offline:   { color: colors.danger.DEFAULT,  label: 'Offline' },
  mining:    { color: colors.accent.DEFAULT,  label: 'Mining', pulse: true },
  leased:    { color: colors.info.DEFAULT,    label: 'Leased' },
  available: { color: colors.success.DEFAULT, label: 'Available' },
  idle:      { color: colors.text.tertiary,   label: 'Idle' },
  error:     { color: colors.danger.DEFAULT,  label: 'Error' },
};

export default function StatusBadge({ status, size = 'md', label: labelOverride }: StatusBadgeProps) {
  const { color, label, pulse } = cfg[status];
  const dot = size === 'sm' ? 'w-1.5 h-1.5' : 'w-2 h-2';
  const text = size === 'sm' ? 'text-xs' : 'text-sm';

  return (
    <span className={`inline-flex items-center gap-1.5 ${text}`} style={{ color }}>
      <span
        className={`${dot} rounded-full ${pulse ? 'animate-pulse' : ''}`}
        style={{ backgroundColor: color, boxShadow: `0 0 6px ${color}50` }}
      />
      {labelOverride ?? label}
    </span>
  );
}
