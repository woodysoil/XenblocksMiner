import { type ReactNode } from 'react';
import { tw } from './tokens';

interface EmptyStateProps {
  icon?: ReactNode;
  title: string;
  description?: string;
  action?: ReactNode;
}

export default function EmptyState({ icon, title, description, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-16">
      {icon && (
        <div className={`w-12 h-12 mb-4 text-[#2a3441] flex items-center justify-center`}>
          {icon}
        </div>
      )}
      <p className={`text-sm font-medium ${tw.textSecondary}`}>{title}</p>
      {description && (
        <p className={`text-xs ${tw.textTertiary} mt-1 max-w-xs text-center`}>{description}</p>
      )}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}
