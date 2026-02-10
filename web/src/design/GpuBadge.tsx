import { tw, colors } from './tokens';

interface GpuBadgeProps {
  name: string;
  memory?: number;
}

export default function GpuBadge({ name, memory }: GpuBadgeProps) {
  const isHighEnd = /4090/i.test(name);
  const isDataCenter = /[AH]100/i.test(name);
  const borderColor = isHighEnd
    ? colors.accent.DEFAULT
    : isDataCenter
      ? colors.warning.DEFAULT
      : 'transparent';

  return (
    <span
      className={`inline-flex items-center gap-1.5 ${tw.surface3} text-xs font-mono px-2 py-1 rounded-md border border-[#2a3441]`}
      style={{ borderLeftWidth: borderColor !== 'transparent' ? 2 : undefined, borderLeftColor: borderColor }}
    >
      <span className={`font-bold ${tw.textPrimary}`}>{name}</span>
      {memory != null && (
        <span className={tw.textSecondary}>{memory} GB</span>
      )}
    </span>
  );
}
