interface SkeletonProps {
  className?: string;
  variant?: 'text' | 'card' | 'circle';
}

const base = 'animate-pulse bg-[#1f2835] rounded';

const variantClass: Record<string, string> = {
  text: 'h-4 w-full rounded',
  card: 'h-32 w-full rounded-[10px]',
  circle: 'rounded-full',
};

export default function Skeleton({ className = '', variant = 'text' }: SkeletonProps) {
  return <div className={`${base} ${variantClass[variant]} ${className}`} />;
}
