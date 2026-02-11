import { colors } from './tokens';

interface ProgressBarProps {
  value: number;
  size?: 'sm' | 'md';
  variant?: 'accent' | 'success' | 'warning' | 'danger';
  showLabel?: boolean;
  label?: string;
}

const fillColor: Record<string, string> = {
  accent: colors.accent.DEFAULT,
  success: colors.success.DEFAULT,
  warning: colors.warning.DEFAULT,
  danger: colors.danger.DEFAULT,
};

export default function ProgressBar({
  value,
  size = 'md',
  variant = 'accent',
  showLabel = false,
  label,
}: ProgressBarProps) {
  const clamped = Math.max(0, Math.min(100, value));
  const height = size === 'sm' ? 4 : 8;

  return (
    <div className="flex items-center gap-2">
      <div
        className="flex-1 overflow-hidden rounded-full"
        style={{ height, backgroundColor: colors.bg.surface3 }}
      >
        <div
          className="h-full rounded-full"
          style={{
            width: `${clamped}%`,
            backgroundColor: fillColor[variant],
            transition: 'width 200ms ease',
          }}
        />
      </div>
      {showLabel && (
        <span className="text-xs shrink-0" style={{ color: colors.text.secondary }}>
          {label ?? `${Math.round(clamped)}%`}
        </span>
      )}
    </div>
  );
}
