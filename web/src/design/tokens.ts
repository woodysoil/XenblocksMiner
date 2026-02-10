/**
 * XenBlocks Design Tokens
 *
 * Single source of truth for the platform's visual language.
 * Import these tokens in all components — never hardcode color/spacing values.
 *
 * Style: Modern dashboard (NiceHash/Grafana-inspired)
 * Theme: Dark-only
 */

// ── Colors ──────────────────────────────────────────────────
export const colors = {
  // Backgrounds (darkest → lightest surface layers)
  bg: {
    base: "#0b0e11",
    surface1: "#141820",
    surface2: "#1a2029",
    surface3: "#1f2835",
    hover: "#252d3a",
  },
  // Borders
  border: {
    default: "#2a3441",
    hover: "#3d4f65",
    active: "#4a6078",
  },
  // Text
  text: {
    primary: "#eaecef",
    secondary: "#848e9c",
    tertiary: "#5e6673",
    inverse: "#0b0e11",
  },
  // Accent — cyan/teal (mining/tech identity)
  accent: {
    DEFAULT: "#22d1ee",
    hover: "#06b6d4",
    muted: "rgba(34,209,238,0.12)",
    glow: "rgba(34,209,238,0.25)",
  },
  // Semantic
  success: { DEFAULT: "#0ecb81", muted: "rgba(14,203,129,0.12)" },
  danger: { DEFAULT: "#f6465d", muted: "rgba(246,70,93,0.12)" },
  warning: { DEFAULT: "#f0b90b", muted: "rgba(240,185,11,0.12)" },
  info: { DEFAULT: "#3b82f6", muted: "rgba(59,130,246,0.12)" },
  // Chart palette (ordered for multi-series)
  chart: ["#22d1ee", "#3b82f6", "#a78bfa", "#f0b90b", "#0ecb81", "#f6465d"],
} as const;

// ── Typography ──────────────────────────────────────────────
export const font = {
  sans: "'Inter', system-ui, -apple-system, sans-serif",
  mono: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
} as const;

// ── Spacing (4px grid) ──────────────────────────────────────
export const space = {
  0: "0",
  1: "4px",
  2: "8px",
  3: "12px",
  4: "16px",
  5: "20px",
  6: "24px",
  8: "32px",
  10: "40px",
  12: "48px",
  16: "64px",
} as const;

// ── Radii ───────────────────────────────────────────────────
export const radius = {
  sm: "4px",
  md: "6px",
  lg: "10px",
  xl: "14px",
  full: "9999px",
} as const;

// ── Tailwind class presets (for consistent usage across components) ────
export const tw = {
  // Cards
  card: "bg-[#141820] border border-[#2a3441] rounded-[10px]",
  cardHover: "hover:border-[#3d4f65] transition-colors duration-150",
  cardInteractive: "bg-[#141820] border border-[#2a3441] rounded-[10px] hover:border-[#3d4f65] transition-colors duration-150 cursor-pointer",

  // Surface layers
  surface1: "bg-[#141820]",
  surface2: "bg-[#1a2029]",
  surface3: "bg-[#1f2835]",

  // Text
  textPrimary: "text-[#eaecef]",
  textSecondary: "text-[#848e9c]",
  textTertiary: "text-[#5e6673]",

  // Badges
  badgeDefault: "text-xs px-2 py-0.5 rounded font-medium",
  badgeSuccess: "text-xs px-2 py-0.5 rounded font-medium bg-[rgba(14,203,129,0.12)] text-[#0ecb81]",
  badgeDanger: "text-xs px-2 py-0.5 rounded font-medium bg-[rgba(246,70,93,0.12)] text-[#f6465d]",
  badgeInfo: "text-xs px-2 py-0.5 rounded font-medium bg-[rgba(59,130,246,0.12)] text-[#3b82f6]",
  badgeWarning: "text-xs px-2 py-0.5 rounded font-medium bg-[rgba(240,185,11,0.12)] text-[#f0b90b]",
  badgeAccent: "text-xs px-2 py-0.5 rounded font-medium bg-[rgba(34,209,238,0.12)] text-[#22d1ee]",

  // Status dots
  dotOnline: "w-2 h-2 rounded-full bg-[#0ecb81] shadow-[0_0_6px_rgba(14,203,129,0.5)]",
  dotOffline: "w-2 h-2 rounded-full bg-[#f6465d]",
  dotActive: "w-2 h-2 rounded-full bg-[#22d1ee] shadow-[0_0_6px_rgba(34,209,238,0.5)]",
  dotIdle: "w-2 h-2 rounded-full bg-[#5e6673]",

  // Buttons
  btnPrimary: "px-4 py-2 rounded-md bg-[#22d1ee] text-[#0b0e11] font-medium text-sm hover:bg-[#06b6d4] transition-colors duration-150",
  btnSecondary: "px-4 py-2 rounded-md bg-[#1f2835] text-[#eaecef] font-medium text-sm border border-[#2a3441] hover:border-[#3d4f65] transition-colors duration-150",
  btnDanger: "px-4 py-2 rounded-md bg-[rgba(246,70,93,0.12)] text-[#f6465d] font-medium text-sm hover:bg-[rgba(246,70,93,0.2)] transition-colors duration-150",

  // Table
  tableHeader: "text-xs text-[#5e6673] uppercase tracking-wider font-medium",
  tableRow: "border-b border-[#1f2835] hover:bg-[#1a2029] transition-colors duration-100",
  tableCell: "px-4 py-3 text-sm text-[#eaecef]",

  // Inputs
  input: "bg-[#0b0e11] border border-[#2a3441] rounded-md px-3 py-2 text-sm text-[#eaecef] placeholder-[#5e6673] focus:border-[#22d1ee] focus:outline-none transition-colors",

  // Section header
  sectionTitle: "text-sm font-semibold text-[#eaecef] uppercase tracking-wide",
} as const;

// ── Chart theme (for recharts) ──────────────────────────────
export const chartTheme = {
  grid: { stroke: "#1f2835", strokeDasharray: "3 3" },
  axis: { stroke: "#2a3441", fontSize: 11, fill: "#5e6673" },
  tooltip: {
    contentStyle: {
      backgroundColor: "#1a2029",
      border: "1px solid #2a3441",
      borderRadius: "6px",
      fontSize: "12px",
      color: "#eaecef",
    },
  },
  areaGradient: { start: "rgba(34,209,238,0.25)", end: "rgba(34,209,238,0)" },
} as const;
