import { colors } from './tokens';

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
      className={`inline-flex items-center gap-1.5 ${`bg-[${colors.bg.surface3}]`} text-xs font-mono px-2 py-1 rounded-md border border-[${colors.border.default}]`}
      style={{ borderLeftWidth: borderColor !== 'transparent' ? 2 : undefined, borderLeftColor: borderColor }}
    >
      <span className={`font-bold text-[${colors.text.primary}]`}>{name}</span>
      {memory != null && (
        <span className={`text-[${colors.text.secondary}]`}>{memory} GB</span>
      )}
    </span>
  );
}
