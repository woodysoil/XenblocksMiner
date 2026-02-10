# XenBlocks Design System

Dark-only dashboard design system inspired by NiceHash and Grafana.
Built with React, TypeScript, and Tailwind CSS.

All source files live under `web/src/design/`. Import from the barrel:

```ts
import { colors, tw, MetricCard, StatusBadge } from '@/design';
```

---

## 1. Design Tokens

Source: `web/src/design/tokens.ts`

### Color Palette

#### Backgrounds (darkest to lightest)

| Token              | Hex       | Usage                          |
|--------------------|-----------|--------------------------------|
| `bg.base`          | `#0b0e11` | Page background, input fields  |
| `bg.surface1`      | `#141820` | Cards, sidebar, dialog panels  |
| `bg.surface2`      | `#1a2029` | Tooltips, hover rows, overlays |
| `bg.surface3`      | `#1f2835` | Nested surfaces, table stripes |
| `bg.hover`         | `#252d3a` | Elevated hover states          |

#### Borders

| Token              | Hex       | Usage                 |
|--------------------|-----------|------------------------|
| `border.default`   | `#2a3441` | Card/input borders     |
| `border.hover`     | `#3d4f65` | Hover border highlight |
| `border.active`    | `#4a6078` | Focus/active borders   |

#### Text

| Token           | Hex       | Usage                           |
|-----------------|-----------|----------------------------------|
| `text.primary`  | `#eaecef` | Headings, values, primary copy   |
| `text.secondary`| `#848e9c` | Labels, descriptions             |
| `text.tertiary` | `#5e6673` | Placeholders, disabled text      |
| `text.inverse`  | `#0b0e11` | Text on accent-colored surfaces  |

#### Accent (Cyan/Teal)

| Token            | Value                      | Usage                      |
|------------------|----------------------------|----------------------------|
| `accent.DEFAULT` | `#22d1ee`                  | Primary accent, active nav |
| `accent.hover`   | `#06b6d4`                  | Button hover state         |
| `accent.muted`   | `rgba(34,209,238,0.12)`    | Badge/tag backgrounds      |
| `accent.glow`    | `rgba(34,209,238,0.25)`    | Box shadow glow effects    |

#### Semantic Colors

| Token               | Solid       | Muted (12% opacity)          |
|----------------------|-------------|------------------------------|
| `success.DEFAULT`    | `#0ecb81`   | `rgba(14,203,129,0.12)`      |
| `danger.DEFAULT`     | `#f6465d`   | `rgba(246,70,93,0.12)`       |
| `warning.DEFAULT`    | `#f0b90b`   | `rgba(240,185,11,0.12)`      |
| `info.DEFAULT`       | `#3b82f6`   | `rgba(59,130,246,0.12)`      |

#### Chart Palette

Six-color series palette ordered for maximum contrast in multi-series charts:

```ts
["#22d1ee", "#3b82f6", "#a78bfa", "#f0b90b", "#0ecb81", "#f6465d"]
```

### Typography

```ts
font.sans  // 'Inter', system-ui, -apple-system, sans-serif
font.mono  // 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace
```

- Body text uses `font.sans`.
- Hash values, addresses, and GPU names use `font.mono`.
- `fontVariantNumeric: 'tabular-nums'` is applied on numeric displays.

### Spacing Grid

4px base grid. Values are exported as CSS strings:

| Token    | Value  |
|----------|--------|
| `space.0`  | `0`    |
| `space.1`  | `4px`  |
| `space.2`  | `8px`  |
| `space.3`  | `12px` |
| `space.4`  | `16px` |
| `space.5`  | `20px` |
| `space.6`  | `24px` |
| `space.8`  | `32px` |
| `space.10` | `40px` |
| `space.12` | `48px` |
| `space.16` | `64px` |

### Border Radii

| Token       | Value    | Usage                    |
|-------------|----------|--------------------------|
| `radius.sm` | `4px`    | Badges, small elements   |
| `radius.md` | `6px`    | Buttons, inputs, tooltips|
| `radius.lg` | `10px`   | Cards                    |
| `radius.xl` | `14px`   | Large panels             |
| `radius.full`| `9999px`| Dots, pills, avatars     |

---

## 2. Tailwind Presets (`tw.*`)

Presets are pre-composed Tailwind class strings. Apply with template literals:

```tsx
<div className={`${tw.card} p-5`}>
```

### Cards

| Preset              | Classes                                                                                     | When to Use                    |
|---------------------|---------------------------------------------------------------------------------------------|--------------------------------|
| `tw.card`           | `bg-[#141820] border border-[#2a3441] rounded-[10px]`                                      | Static card container          |
| `tw.cardHover`      | `hover:border-[#3d4f65] transition-colors duration-150`                                     | Add to `tw.card` for hover     |
| `tw.cardInteractive`| `tw.card` + `tw.cardHover` + `cursor-pointer`                                              | Clickable cards                |

### Surfaces

| Preset         | Hex       | Usage                        |
|----------------|-----------|------------------------------|
| `tw.surface1`  | `#141820` | Sidebar, card backgrounds    |
| `tw.surface2`  | `#1a2029` | Nested panels, hover rows    |
| `tw.surface3`  | `#1f2835` | Deepest nested surfaces      |

### Text

| Preset            | Color     | Usage                     |
|-------------------|-----------|---------------------------|
| `tw.textPrimary`  | `#eaecef` | Headings, values          |
| `tw.textSecondary`| `#848e9c` | Labels, descriptions      |
| `tw.textTertiary` | `#5e6673` | Hints, disabled, muted    |

### Badges

All badges share base sizing: `text-xs px-2 py-0.5 rounded font-medium`.

| Preset             | Background             | Text        |
|--------------------|------------------------|-------------|
| `tw.badgeDefault`  | (none)                 | (none)      |
| `tw.badgeSuccess`  | `success.muted`        | `#0ecb81`   |
| `tw.badgeDanger`   | `danger.muted`         | `#f6465d`   |
| `tw.badgeInfo`     | `info.muted`           | `#3b82f6`   |
| `tw.badgeWarning`  | `warning.muted`        | `#f0b90b`   |
| `tw.badgeAccent`   | `accent.muted`         | `#22d1ee`   |

### Status Dots

| Preset          | Visual                              |
|-----------------|--------------------------------------|
| `tw.dotOnline`  | Green `#0ecb81` with glow shadow     |
| `tw.dotOffline` | Red `#f6465d`, no glow               |
| `tw.dotActive`  | Cyan `#22d1ee` with glow shadow      |
| `tw.dotIdle`    | Tertiary `#5e6673`, no glow          |

All dots are `w-2 h-2 rounded-full`.

### Buttons

| Preset            | Style                                       |
|-------------------|----------------------------------------------|
| `tw.btnPrimary`   | Solid cyan bg, dark text, cyan hover         |
| `tw.btnSecondary` | Surface3 bg, light text, border highlight    |
| `tw.btnDanger`    | Translucent red bg, red text, deeper on hover|

```tsx
<button className={tw.btnPrimary}>Save</button>
<button className={tw.btnSecondary}>Cancel</button>
<button className={tw.btnDanger}>Delete</button>
```

### Table

| Preset           | Usage                                     |
|------------------|-------------------------------------------|
| `tw.tableHeader` | `<th>` — uppercase, tertiary, xs, tracked |
| `tw.tableRow`    | `<tr>` — bottom border, hover bg          |
| `tw.tableCell`   | `<td>` — sm text, primary color, padded   |

### Input

```tsx
<input className={tw.input} placeholder="Search..." />
```

Dark bg, border, cyan focus ring, placeholder in tertiary.

### Section Title

```tsx
<h2 className={tw.sectionTitle}>Fleet Status</h2>
```

Uppercase, sm, semibold, tracked.

---

## 3. Component Reference

### MetricCard

Top-level KPI display with optional colored top border and glow.

**Source:** `web/src/design/MetricCard.tsx`

```ts
interface MetricCardProps {
  label: string;
  value: string | number;
  delta?: string;                // "+12%" or "-3.5%" — auto-colored
  icon?: ReactNode;
  variant?: 'default' | 'accent' | 'success' | 'danger' | 'info' | 'warning';
  loading?: boolean;
}
```

**Usage:**

```tsx
<MetricCard
  label="Total Hashrate"
  value="1,842 H/s"
  delta="+12.4%"
  variant="accent"
  icon={<HashIcon />}
/>

<MetricCard label="Miners" value={42} loading />
```

**Variants:** Non-default variants render a 2px colored top border and a subtle box-shadow glow matching the variant color.

**States:**
- `loading={true}` replaces value with `<Skeleton>` placeholders.
- `delta` text auto-colors: `+` prefix renders green (`success`), `-` prefix renders red (`danger`), otherwise secondary text.

**Accessibility:** Uses `fontVariantNumeric: 'tabular-nums'` to prevent layout shift on numeric changes.

---

### StatusBadge

Inline status indicator with colored dot and label.

**Source:** `web/src/design/StatusBadge.tsx`

```ts
type Status = 'online' | 'offline' | 'mining' | 'leased' | 'available' | 'idle' | 'error';

interface StatusBadgeProps {
  status: Status;
  size?: 'sm' | 'md';    // default: 'md'
  label?: string;         // override default label text
}
```

**Usage:**

```tsx
<StatusBadge status="mining" />
<StatusBadge status="online" size="sm" />
<StatusBadge status="offline" label="Unreachable" />
```

**Status mapping:**

| Status      | Color           | Pulse |
|-------------|-----------------|-------|
| `online`    | `success`       | no    |
| `offline`   | `danger`        | no    |
| `mining`    | `accent` (cyan) | yes   |
| `leased`    | `info`          | no    |
| `available` | `success`       | no    |
| `idle`      | `tertiary`      | no    |
| `error`     | `danger`        | no    |

**Accessibility:** Dot receives a subtle `box-shadow` glow to reinforce color meaning beyond hue alone. The `mining` status uses `animate-pulse` to convey active processing.

---

### GpuBadge

Monospace GPU identifier badge with tier-based accent border.

**Source:** `web/src/design/GpuBadge.tsx`

```ts
interface GpuBadgeProps {
  name: string;
  memory?: number;   // GB
}
```

**Usage:**

```tsx
<GpuBadge name="RTX 4090" memory={24} />
<GpuBadge name="A100" memory={80} />
<GpuBadge name="RTX 3080" memory={10} />
```

**Tier borders:**
- Names matching `/4090/i` receive a cyan (`accent`) left border.
- Names matching `/[AH]100/i` receive a yellow (`warning`) left border.
- All others get no accent border.

---

### ChartCard

Container for chart visualizations with a header section.

**Source:** `web/src/design/ChartCard.tsx`

```ts
interface ChartCardProps {
  title: string;
  subtitle?: string;
  action?: ReactNode;    // slot for time-range selector, toggle, etc.
  children: ReactNode;   // the chart content
}
```

**Usage:**

```tsx
<ChartCard
  title="Hashrate (24h)"
  subtitle="Average: 1,842 H/s"
  action={<TimeRangeSelector />}
>
  <LWChart data={points} height={260} />
</ChartCard>
```

**Layout:** Header sits in `px-5 pt-5 pb-3`. Chart area uses `px-3 pb-4` for reduced horizontal padding to maximize chart width.

---

### LWChart

Lightweight Charts (TradingView) area chart wrapper with auto-resize and visible window control.

**Source:** `web/src/design/LWChart.tsx`

```ts
interface LWDataPoint {
  time: UTCTimestamp;
  value: number;
}

interface LWChartProps {
  data: LWDataPoint[];
  height?: number;              // default: 260
  formatValue?: (v: number) => string;
  visibleWindow?: number;       // seconds, 0 = show all (default)
}
```

**Usage:**

```tsx
import { type UTCTimestamp } from 'lightweight-charts';

const points: LWDataPoint[] = [
  { time: 1700000000 as UTCTimestamp, value: 1842 },
  { time: 1700003600 as UTCTimestamp, value: 1956 },
];

<LWChart data={points} height={300} formatValue={(v) => `${v} H/s`} />
<LWChart data={points} visibleWindow={86400} />  {/* last 24h */}
```

**Behavior:**
- Chart is lazily created on first data update (deferred until container has nonzero width).
- A `ResizeObserver` keeps chart width in sync with container.
- Cleanup on unmount disconnects the observer and removes the chart instance.
- `visibleWindow` auto-scrolls to show the last N seconds; set to `0` or omit to fit all data.

**Theme:** Transparent background (inherits card), accent-colored area fill with gradient from `accent.glow` to transparent, grid lines use `bg.surface3`, axis text uses `text.secondary`.

---

### EmptyState

Zero-data placeholder for tables and lists.

**Source:** `web/src/design/EmptyState.tsx`

```ts
interface EmptyStateProps {
  icon?: ReactNode;
  title: string;
  description?: string;
  action?: ReactNode;
}
```

**Usage:**

```tsx
<EmptyState
  icon={<ServerIcon />}
  title="No miners found"
  description="Connect a miner to get started."
  action={<button className={tw.btnPrimary}>Add Miner</button>}
/>
```

**Layout:** Vertically centered with `py-16`. Icon renders at `48x48` in border-default color. Description is constrained to `max-w-xs` and centered.

---

### Skeleton

Animated loading placeholder.

**Source:** `web/src/design/Skeleton.tsx`

```ts
interface SkeletonProps {
  className?: string;
  variant?: 'text' | 'card' | 'circle';   // default: 'text'
}
```

**Usage:**

```tsx
<Skeleton />                           {/* full-width text line */}
<Skeleton className="h-7 w-24" />      {/* custom dimensions */}
<Skeleton variant="card" />            {/* 128px tall card shape */}
<Skeleton variant="circle" className="w-10 h-10" />
```

**Variants:**

| Variant  | Default Shape            |
|----------|--------------------------|
| `text`   | `h-4 w-full rounded`     |
| `card`   | `h-32 w-full rounded-lg` |
| `circle` | `rounded-full`           |

All variants use `animate-pulse` with `bg-[#1f2835]`.

---

### ViewToggle

Grid/list view switcher with inline SVG icons.

**Source:** `web/src/design/ViewToggle.tsx`

```ts
type ViewMode = 'grid' | 'list';

interface ViewToggleProps {
  value: ViewMode;
  onChange: (mode: ViewMode) => void;
}
```

**Usage:**

```tsx
const [view, setView] = useState<ViewMode>('grid');

<ViewToggle value={view} onChange={setView} />
```

**Accessibility:** Wraps buttons in `role="group"` with `aria-label="View mode"`. Each button has individual `aria-label` and `title` attributes.

---

### ConfirmDialog

Modal confirmation dialog built on Radix UI `AlertDialog`.

**Source:** `web/src/design/ConfirmDialog.tsx`

```ts
interface ConfirmDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description: string;
  confirmLabel?: string;          // default: "Confirm"
  variant?: 'danger' | 'primary'; // default: 'primary'
  onConfirm: () => void;
}
```

**Usage:**

```tsx
<ConfirmDialog
  open={showDelete}
  onOpenChange={setShowDelete}
  title="Remove Miner"
  description="This will permanently remove the miner from your fleet."
  confirmLabel="Remove"
  variant="danger"
  onConfirm={handleDelete}
/>
```

**Variants:**
- `primary` renders the confirm button with `tw.btnPrimary` (cyan).
- `danger` renders it with `tw.btnDanger` (red).

**Accessibility:** Uses Radix `AlertDialog` primitives, providing native focus trapping, escape-to-close, and `aria-labelledby`/`aria-describedby` via `AlertDialog.Title`/`AlertDialog.Description`. Focus-visible outlines are explicitly styled on both Cancel and Action buttons.

**Animation:** Entry/exit uses `animate-in`/`animate-out` with fade and zoom-95 transforms. Overlay applies `backdrop-blur-sm`.

---

### HashText

Truncated hash/address display with optional copy-to-clipboard.

**Source:** `web/src/design/HashText.tsx`

```ts
interface HashTextProps {
  text: string;
  chars?: number;       // leading chars to show (default: 8)
  mono?: boolean;       // monospace font (default: true)
  copyable?: boolean;   // show copy button on hover (default: false)
}
```

**Usage:**

```tsx
<HashText text="0xabc123def456789..." copyable />
<HashText text="0xabc123def456789..." chars={12} />
```

**Behavior:**
- Truncates to `{chars}...{last 4}` when text exceeds `chars + 4` length.
- Full text is available via `title` attribute on hover.
- Copy button fades in on group hover, shows a checkmark for 1.5s after copy.

**Accessibility:** Copy button has `aria-label="Copy"`. Full text in `title` is accessible to screen readers.

---

### Pagination

Page navigation with ellipsis compression.

**Source:** `web/src/components/Pagination.tsx`

```ts
interface PaginationProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
}
```

**Usage:**

```tsx
<Pagination currentPage={3} totalPages={12} onPageChange={setPage} />
```

**Behavior:**
- Returns `null` when `totalPages <= 1`.
- Shows up to 7 page buttons. Pages beyond the visible range collapse to `...` ellipsis.
- Current page is highlighted with an accent-muted background and accent text.
- Prev/Next buttons disable at boundaries.
- Renders a "Page X of Y" indicator to the right.

---

## 4. Layout System

**Source:** `web/src/components/Layout.tsx`

### Structure

```
+-------------------+------------------------------------------+
|     Sidebar       |              Header (sticky)             |
|     (fixed)       +------------------------------------------+
|     w-56          |                                          |
|                   |              Main Content                |
|  - Logo           |              (scrollable)                |
|  - Public nav     |              p-4 sm:p-6                  |
|  - Wallet nav     |                                          |
|  - Status bar     |                                          |
+-------------------+------------------------------------------+
```

### Sidebar

- Fixed position, `w-56` (224px), full height.
- Background: `surface1` (`#141820`), right border `border.default`.
- Contains logo, two nav sections separated by a labeled divider ("Wallet"), and a connection status bar at the bottom.
- Bottom area has a gradient fade from transparent to `surface1` to indicate scrollable overflow.

### Navigation

Two nav groups:

**Public:**
| Route           | Label        |
|-----------------|--------------|
| `/`             | Overview     |
| `/monitoring`   | Monitoring   |
| `/marketplace`  | Marketplace  |

**Wallet-gated:**
| Route       | Label    |
|-------------|----------|
| `/provider` | Provider |
| `/renter`   | Renter   |
| `/account`  | Account  |

Active nav items display:
- `bg-[#22d1ee]/8` background tint
- Cyan text color
- A 3px rounded cyan indicator bar on the left edge

Inactive items use `text.secondary` with hover transitions to `text.primary`.

### Header

- Sticky, `h-14`, `backdrop-blur-md`, semi-transparent `surface1` background.
- Left: page title (dynamic via route).
- Right: search input (hidden on mobile), wallet connect button, notification bell.

### Content Area

- `flex-1 overflow-y-auto` with `p-4 sm:p-6`.
- Route content renders via `<Outlet />` wrapped in `animate-page-enter` for entry animation.

### Mobile Responsive Behavior

| Breakpoint | Behavior                                                    |
|------------|-------------------------------------------------------------|
| `< md`     | Sidebar hidden off-screen (`-translate-x-full`). Hamburger button in header. Tap opens sidebar with slide transition + overlay backdrop (`bg-black/50`). |
| `>= md`    | Sidebar always visible. Hamburger hidden. Main content offset by `ml-56`. Search input visible. |

Mobile sidebar is `z-40`, overlay is `z-30`. Clicking overlay or any nav link closes the sidebar.

---

## 5. Theming

### Dark-Only

The system is dark-only. There is no light theme or `prefers-color-scheme` toggling. The `<html>` element does not require a `dark` class.

### Color Token Rules

1. **Never hardcode hex values in components.** Import from `colors` or use `tw.*` presets.
2. **Semantic colors always come in pairs:** a solid value for text/borders and a `muted` value (12% opacity) for backgrounds.
3. **Surface layering:** `bg.base` < `bg.surface1` < `bg.surface2` < `bg.surface3` < `bg.hover`. Each layer adds elevation.

### Tailwind Purge-Safe Patterns

Dynamic color values (computed at runtime) must use inline `style` attributes. Tailwind cannot detect dynamically constructed class names during its purge pass.

**Correct** -- dynamic color via inline style:
```tsx
// Color determined at runtime — use style
<span style={{ color: variantColor[variant] }}>Text</span>
<div style={{ borderTopColor: variantColor[variant], boxShadow: variantGlow[variant] }}>
```

**Correct** -- static class for fixed color:
```tsx
// Color is known at build time — use class
<span className="text-[#22d1ee]">Accent</span>
<div className={tw.badgeSuccess}>OK</div>
```

**Incorrect** -- dynamic Tailwind class:
```tsx
// BROKEN: Tailwind purges this because the full class is never in source
const color = variant === 'danger' ? 'text-[#f6465d]' : 'text-[#22d1ee]';
<span className={color}>Text</span>
```

This pattern is safe when both branches appear as complete static strings in the source file (Tailwind scans for complete class matches). But when the string is constructed programmatically or comes from a variable lookup, use `style` instead.

### Chart Theming

The `chartTheme` object provides a consistent theme for Recharts:

```ts
import { chartTheme } from '@/design';

<CartesianGrid {...chartTheme.grid} />
<XAxis {...chartTheme.axis} />
<Tooltip {...chartTheme.tooltip} />
```

For Lightweight Charts, `LWChart` applies the theme internally. No external configuration needed.

---

## 6. Icons

All icons are inline SVGs. No icon library is used.

### Conventions

| Property            | Value              |
|---------------------|--------------------|
| Default size        | `20x20` (nav), `18x18` (header), `16x16` (toggles) |
| `fill`              | `none` for stroke icons, `currentColor` for filled   |
| `stroke`            | `currentColor`     |
| `strokeWidth`       | `1.5`              |
| `strokeLinecap`     | `round` (where applicable) |

### Pattern

Icons inherit color from their parent via `currentColor`. This integrates with Tailwind text-color utilities and transition classes:

```tsx
<span className="text-[#848e9c] hover:text-[#eaecef] transition-colors">
  <svg width="20" height="20" viewBox="0 0 20 20"
       fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="..." />
  </svg>
</span>
```

### Common Icons in Use

| Icon         | Context          | Style          |
|--------------|------------------|----------------|
| Grid (4-rect)| ViewToggle, nav | `fill` based   |
| List (3-bar) | ViewToggle       | `fill` based   |
| Bell         | Header notif     | `stroke` based |
| Close (X)    | Mobile sidebar   | `stroke` based |
| Hamburger    | Mobile header    | `stroke` based |
| Checkmark    | HashText copy    | `fill` based   |
| Copy         | HashText         | `fill` based   |
| Diamond      | Logo             | `fill` + `stroke` combined |

For small utility icons (copy, check) inside interactive components, use `w-3.5 h-3.5`. For navigation icons, use `w-5 h-5` with a flex centering wrapper.
