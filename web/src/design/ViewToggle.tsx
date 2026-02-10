import { tw } from "./tokens";

type ViewMode = "grid" | "list";

interface ViewToggleProps {
  value: ViewMode;
  onChange: (mode: ViewMode) => void;
}

export default function ViewToggle({ value, onChange }: ViewToggleProps) {
  const base = "p-1.5 rounded transition-colors";
  const active = "bg-[#22d1ee]/15 text-[#22d1ee]";
  const inactive = "text-[#5e6673] hover:text-[#848e9c]";

  return (
    <div className="flex items-center gap-0.5 bg-[#0b0e11] rounded-md p-0.5 border border-[#2a3441]" role="group" aria-label="View mode">
      <button
        onClick={() => onChange("grid")}
        className={`${base} ${value === "grid" ? active : inactive}`}
        title="Grid view"
        aria-label="Grid view"
      >
        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
          <rect x="1" y="1" width="6" height="6" rx="1" />
          <rect x="9" y="1" width="6" height="6" rx="1" />
          <rect x="1" y="9" width="6" height="6" rx="1" />
          <rect x="9" y="9" width="6" height="6" rx="1" />
        </svg>
      </button>
      <button
        onClick={() => onChange("list")}
        className={`${base} ${value === "list" ? active : inactive}`}
        title="List view"
        aria-label="List view"
      >
        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
          <rect x="1" y="1.5" width="14" height="3" rx="1" />
          <rect x="1" y="6.5" width="14" height="3" rx="1" />
          <rect x="1" y="11.5" width="14" height="3" rx="1" />
        </svg>
      </button>
    </div>
  );
}

export type { ViewMode };
