import { colors } from './tokens';

interface SkeletonProps {
  className?: string;
  variant?: 'text' | 'card' | 'circle' | 'chart';
}

const base = 'animate-pulse bg-[#1f2835] rounded';

const variantClass: Record<string, string> = {
  text: 'h-4 w-full rounded',
  card: 'h-32 w-full rounded-[10px]',
  circle: 'rounded-full',
  chart: 'h-48 w-full rounded-[10px] overflow-hidden',
};

function ChartWave() {
  return (
    <svg
      className="absolute inset-0 w-full h-full opacity-[0.15]"
      preserveAspectRatio="none"
      viewBox="0 0 400 200"
    >
      <path
        d="M0 140 Q50 100 100 120 T200 100 T300 130 T400 90"
        fill="none"
        stroke={colors.text.tertiary}
        strokeWidth="2"
      />
      <path
        d="M0 160 Q80 130 160 150 T320 120 T400 140"
        fill="none"
        stroke={colors.text.tertiary}
        strokeWidth="1.5"
        strokeDasharray="6 4"
      />
    </svg>
  );
}

export default function Skeleton({ className = '', variant = 'text' }: SkeletonProps) {
  if (variant === 'chart') {
    return (
      <div className={`${base} ${variantClass.chart} relative ${className}`}>
        <ChartWave />
      </div>
    );
  }
  return <div className={`${base} ${variantClass[variant]} ${className}`} />;
}
