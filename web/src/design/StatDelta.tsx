import { colors } from './tokens';

interface StatDeltaProps {
  value: number;
  suffix?: string;
}

export default function StatDelta({ value, suffix = '%' }: StatDeltaProps) {
  const positive = value >= 0;
  const color = positive ? colors.success.DEFAULT : colors.danger.DEFAULT;
  const arrow = positive ? '\u25B2' : '\u25BC';
  const sign = positive ? '+' : '';

  return (
    <span className="inline-flex items-center gap-0.5 text-xs font-medium" style={{ color }}>
      <span style={{ fontSize: 8 }}>{arrow}</span>
      {sign}{value.toFixed(1)}{suffix}
    </span>
  );
}
