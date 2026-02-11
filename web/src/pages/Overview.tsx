import { useState, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "../api";
import { tw, colors } from "../design/tokens";
import Skeleton from "../design/Skeleton";
import MetricCard from "../design/MetricCard";
import EmptyState from "../design/EmptyState";
import Pagination from "../components/Pagination";
import { formatHashrate, timeAgo } from "../utils/format";

interface OverviewStats {
  total_workers: number;
  online_workers: number;
  total_hashrate: number;
  total_blocks: number;
  blocks_24h: number;
  active_leases: number;
  total_settled: number;
  platform_revenue: number;
}

interface ActivityItem {
  type: string;
  timestamp: number | string;
  details: Record<string, unknown>;
}

interface ActivityResponse {
  items: ActivityItem[];
  total: number;
  page: number;
  limit: number;
  total_pages: number;
}

interface NetworkInfo {
  difficulty: number;
  total_workers: number;
  total_blocks: number;
  chain_blocks: number;
}

// ── Event styling ──────────────────────────────────────────

const eventDotColor: Record<string, string> = {
  block: colors.accent.DEFAULT,
  block_found: colors.accent.DEFAULT,
  lease_started: colors.info.DEFAULT,
  lease_completed: colors.success.DEFAULT,
  worker_registered: colors.warning.DEFAULT,
};

const eventBadgeVariant: Record<string, string> = {
  block: tw.badgeAccent,
  block_found: tw.badgeAccent,
  lease_started: tw.badgeInfo,
  lease_completed: tw.badgeSuccess,
  worker_registered: tw.badgeWarning,
};

const eventLabel = (a: ActivityItem): string => {
  const wid = a.details?.worker_id ? String(a.details.worker_id) : "";
  switch (a.type) {
    case "block":
    case "block_found": return `Block mined by ${wid || "unknown"}`;
    case "lease_started": return `Lease started${wid ? ` — ${wid}` : ""}`;
    case "lease_completed": return `Lease completed${wid ? ` — ${wid}` : ""}`;
    default: return `${a.type.replace(/_/g, " ")}${wid ? ` — ${wid}` : ""}`;
  }
};

const eventTagLabel = (type: string): string => {
  switch (type) {
    case "block":
    case "block_found": return "Block";
    case "lease_started": return "Lease";
    case "lease_completed": return "Done";
    case "worker_registered": return "Worker";
    default: return type.split("_")[0];
  }
};

function normalizeActivity(d: ActivityResponse | ActivityItem[]): {
  items: ActivityItem[];
  totalPages: number;
} {
  if (Array.isArray(d)) return { items: d, totalPages: 1 };
  return { items: d.items || [], totalPages: d.total_pages || 1 };
}

// ── Time grouping ──────────────────────────────────────────

function timePeriod(ts: number | string): string {
  const ms = typeof ts === "number" ? (ts > 1e12 ? ts : ts * 1000) : new Date(ts).getTime();
  const diff = Date.now() - ms;
  if (diff < 86_400_000) return "Today";
  if (diff < 172_800_000) return "Yesterday";
  return "This Week";
}

function groupByPeriod(items: ActivityItem[]): { period: string; items: ActivityItem[] }[] {
  const order = ["Today", "Yesterday", "This Week"];
  const map = new Map<string, ActivityItem[]>();
  for (const item of items) {
    const p = timePeriod(item.timestamp);
    if (!map.has(p)) map.set(p, []);
    map.get(p)!.push(item);
  }
  return order.filter((p) => map.has(p)).map((period) => ({ period, items: map.get(period)! }));
}

// ── Inline SVG Icons ───────────────────────────────────────

const IconHashrate = (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke={colors.accent.DEFAULT} strokeWidth="1.5">
    <path d="M10 2v4M10 14v4M4 8l2.5 4M13.5 8l2.5 4M6.5 12h7" />
    <circle cx="10" cy="10" r="8" strokeDasharray="2 2" opacity="0.3" />
  </svg>
);

const IconMiners = (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
    <circle cx="10" cy="10" r="4" fill={colors.success.DEFAULT} opacity="0.2" />
    <circle cx="10" cy="10" r="2" fill={colors.success.DEFAULT} />
  </svg>
);

const IconBlocks = (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke={colors.info.DEFAULT} strokeWidth="1.5">
    <rect x="3" y="3" width="14" height="14" rx="2" />
    <path d="M3 8h14M8 3v14" />
  </svg>
);

const IconRevenue = (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke={colors.warning.DEFAULT} strokeWidth="1.5">
    <circle cx="10" cy="10" r="7" />
    <path d="M10 6v8M8 8h4M8 12h4" />
  </svg>
);

const IconNetwork = (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke={colors.accent.DEFAULT} strokeWidth="1.5" strokeLinecap="round">
    <circle cx="10" cy="4" r="2" /><circle cx="4" cy="14" r="2" /><circle cx="16" cy="14" r="2" />
    <path d="M10 6v3M8 11l-3 2M12 11l3 2" />
  </svg>
);

const IconEmptyActivity = (
  <svg width="48" height="48" viewBox="0 0 48 48" fill="none" stroke={colors.border.default} strokeWidth="1.5" strokeLinecap="round">
    <rect x="8" y="6" width="32" height="36" rx="4" />
    <path d="M16 16h16M16 24h10M16 32h12" opacity="0.5" />
    <circle cx="36" cy="36" r="8" fill={colors.bg.surface1} stroke={colors.border.default} />
    <path d="M33 36h6M36 33v6" stroke={colors.text.tertiary} />
  </svg>
);

// ── Network sparkline (simple difficulty trend visual) ─────

function DifficultySparkline({ value }: { value: number }) {
  // Deterministic pseudo-sparkline derived from the difficulty value
  const seed = value % 10000;
  const points: number[] = [];
  let v = 40;
  for (let i = 0; i < 12; i++) {
    v = Math.max(8, Math.min(52, v + ((((seed * (i + 1) * 7) % 19) - 9) * 1.5)));
    points.push(v);
  }
  const step = 80 / (points.length - 1);
  const d = points.map((y, i) => `${i === 0 ? "M" : "L"}${i * step},${y}`).join(" ");
  return (
    <svg width="80" height="56" viewBox="0 0 80 56" className="opacity-60">
      <defs>
        <linearGradient id="spark-g" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={colors.accent.DEFAULT} stopOpacity="0.25" />
          <stop offset="100%" stopColor={colors.accent.DEFAULT} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={`${d} L80,56 L0,56 Z`} fill="url(#spark-g)" />
      <path d={d} fill="none" stroke={colors.accent.DEFAULT} strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

// ── Detail panel for expanded activity item ────────────────

function ActivityDetails({ item }: { item: ActivityItem }) {
  const entries = Object.entries(item.details).filter(
    ([k]) => k !== "worker_id",
  );
  if (entries.length === 0) {
    return (
      <div className={`text-xs ${tw.textTertiary} pl-6 pb-2`}>
        No additional details
      </div>
    );
  }
  return (
    <div className={`grid grid-cols-2 gap-x-6 gap-y-1 pl-6 pb-2`}>
      {entries.map(([k, v]) => (
        <div key={k} className="flex items-baseline gap-2 text-xs">
          <span className={tw.textTertiary}>{k.replace(/_/g, " ")}:</span>
          <span className={tw.textSecondary}>{String(v)}</span>
        </div>
      ))}
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────

export default function Overview() {
  const [activityPage, setActivityPage] = useState(1);
  const [expandedIdx, setExpandedIdx] = useState<string | null>(null);

  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ["overview", "stats"],
    queryFn: () => apiFetch<OverviewStats>("/api/overview/stats"),
    refetchInterval: 10_000,
  });

  const { data: activityData, isLoading: activityLoading } = useQuery({
    queryKey: ["overview", "activity", activityPage],
    queryFn: () =>
      apiFetch<ActivityResponse | ActivityItem[]>(
        `/api/overview/activity?page=${activityPage}&limit=50`,
      ),
    select: normalizeActivity,
    refetchInterval: 10_000,
  });

  const activity = activityData?.items ?? [];
  const activityTotalPages = activityData?.totalPages ?? 1;

  const { data: network, isLoading: networkLoading } = useQuery({
    queryKey: ["overview", "network"],
    queryFn: () => apiFetch<NetworkInfo>("/api/overview/network"),
    refetchInterval: 10_000,
  });

  const toggleExpand = useCallback(
    (key: string) => setExpandedIdx((prev) => (prev === key ? null : key)),
    [],
  );

  // ── Card definitions ─────────────────────────────────────

  const cards: {
    label: string;
    value: string | number;
    variant: "accent" | "success" | "info" | "warning";
    icon: JSX.Element;
  }[] = [
    {
      label: "Total Hashrate",
      value: stats ? formatHashrate(stats.total_hashrate) : "\u2014",
      variant: "accent",
      icon: IconHashrate,
    },
    {
      label: "Active Miners",
      value: stats?.online_workers ?? "\u2014",
      variant: "success",
      icon: IconMiners,
    },
    {
      label: "Blocks (24h)",
      value: stats?.blocks_24h ?? "\u2014",
      variant: "info",
      icon: IconBlocks,
    },
    {
      label: "Revenue",
      value: stats ? `${stats.platform_revenue.toFixed(2)} XNM` : "\u2014",
      variant: "warning",
      icon: IconRevenue,
    },
  ];

  // ── Network items ────────────────────────────────────────

  const networkRows = network
    ? [
        { label: "Difficulty", value: network.difficulty.toLocaleString() },
        { label: "Total Workers", value: network.total_workers.toLocaleString() },
        { label: "Total Blocks", value: network.total_blocks.toLocaleString() },
        { label: "Chain Blocks", value: network.chain_blocks.toLocaleString() },
      ]
    : null;

  // ── Grouped activity ─────────────────────────────────────

  const groups = activity.length > 0 ? groupByPeriod(activity) : [];

  return (
    <div className="space-y-6">
      {/* ── Stats row ─────────────────────────────────────── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {cards.map((c) => (
          <MetricCard
            key={c.label}
            label={c.label}
            value={c.value}
            icon={c.icon}
            variant={c.variant}
            loading={statsLoading && !stats}
          />
        ))}
      </div>

      {/* ── Activity + Network ────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Activity Feed */}
        <div className={`lg:col-span-3 ${tw.card} flex flex-col`}>
          <div className="flex items-center justify-between px-5 pt-4 pb-3">
            <h3 className={`text-sm font-semibold ${tw.textPrimary}`}>Recent Activity</h3>
            <span className={`text-xs ${tw.textTertiary}`}>Last 24h</span>
          </div>

          <div className="px-5 pb-4 flex-1 max-h-[440px] overflow-y-auto">
            {activityLoading && !activityData ? (
              /* Skeleton loader matching final layout */
              <div className="space-y-0">
                {Array.from({ length: 6 }).map((_, i) => (
                  <div
                    key={i}
                    className={`flex items-center gap-3 py-3 ${
                      i < 5 ? "border-b border-[#1f2835]" : ""
                    }`}
                  >
                    <Skeleton variant="circle" className="w-2.5 h-2.5 shrink-0" />
                    <Skeleton className="h-4 flex-1" />
                    <Skeleton className="h-4 w-12 shrink-0" />
                    <Skeleton className="h-3 w-14 shrink-0" />
                  </div>
                ))}
              </div>
            ) : activity.length === 0 ? (
              <EmptyState
                icon={IconEmptyActivity}
                title="No activity yet"
                description="Mining events, leases, and worker registrations will appear here as they happen."
              />
            ) : (
              groups.map((group) => (
                <div key={group.period}>
                  <div
                    className={`text-[10px] uppercase tracking-widest font-semibold ${tw.textTertiary} mt-3 mb-1 first:mt-0`}
                  >
                    {group.period}
                  </div>
                  {group.items.map((a, i) => {
                    const key = `${group.period}-${i}`;
                    const isExpanded = expandedIdx === key;
                    return (
                      <div key={key}>
                        <button
                          type="button"
                          onClick={() => toggleExpand(key)}
                          className={`w-full flex items-center gap-3 py-2.5 border-b border-[#1f2835] text-sm text-left transition-colors duration-100 hover:bg-[#252d3a] rounded ${
                            isExpanded ? "bg-[#1a2029]" : ""
                          }`}
                        >
                          {/* Dot + timeline connector */}
                          <div className="relative shrink-0 flex items-center justify-center w-2.5">
                            {i < group.items.length - 1 && (
                              <div
                                className="absolute top-3 bottom-[-13px] left-1/2 -translate-x-1/2 w-px"
                                style={{ backgroundColor: colors.bg.surface3 }}
                              />
                            )}
                            <span
                              className="w-2.5 h-2.5 rounded-full shrink-0 relative z-10"
                              style={{
                                backgroundColor: eventDotColor[a.type] || colors.text.tertiary,
                                boxShadow: eventDotColor[a.type]
                                  ? `0 0 6px ${eventDotColor[a.type]}40`
                                  : undefined,
                              }}
                            />
                          </div>

                          <span className={`flex-1 truncate ${tw.textPrimary}`}>
                            {eventLabel(a)}
                          </span>

                          {/* Type badge */}
                          <span
                            className={`${
                              eventBadgeVariant[a.type] || tw.badgeDefault
                            } shrink-0 hidden sm:inline-block`}
                          >
                            {eventTagLabel(a.type)}
                          </span>

                          <span
                            className={`text-xs ${tw.textTertiary} tabular-nums shrink-0 w-16 text-right`}
                          >
                            {timeAgo(a.timestamp).text}
                          </span>
                        </button>

                        {isExpanded && <ActivityDetails item={a} />}
                      </div>
                    );
                  })}
                </div>
              ))
            )}
          </div>

          <Pagination
            currentPage={activityPage}
            totalPages={activityTotalPages}
            onPageChange={setActivityPage}
          />
        </div>

        {/* ── Network Status ──────────────────────────────── */}
        <div className={`lg:col-span-2 ${tw.card} p-5 flex flex-col`}>
          <div className="flex items-center gap-2 mb-4">
            {IconNetwork}
            <h3 className={`text-sm font-semibold ${tw.textPrimary}`}>Network Status</h3>
          </div>

          {/* Difficulty sparkline */}
          {network && (
            <div
              className="rounded-lg mb-4 px-4 pt-3 pb-2 flex items-end justify-between"
              style={{ backgroundColor: colors.bg.surface2 }}
            >
              <div>
                <span className={`text-[10px] uppercase tracking-wider ${tw.textTertiary}`}>
                  Difficulty
                </span>
                <div className={`text-lg font-bold ${tw.textPrimary} tabular-nums`}>
                  {network.difficulty.toLocaleString()}
                </div>
              </div>
              <DifficultySparkline value={network.difficulty} />
            </div>
          )}
          {networkLoading && !network && (
            <Skeleton className="h-20 w-full rounded-lg mb-4" />
          )}

          {/* Remaining network rows */}
          {networkLoading && !network
            ? Array.from({ length: 3 }).map((_, i) => (
                <div
                  key={i}
                  className={`flex justify-between items-center py-3 ${
                    i < 2 ? "border-b border-[#1f2835]/60" : ""
                  }`}
                >
                  <Skeleton className="h-3 w-20" />
                  <Skeleton className="h-4 w-16" />
                </div>
              ))
            : networkRows
              ? networkRows
                  .filter((r) => r.label !== "Difficulty")
                  .map((item, i, arr) => (
                    <div
                      key={item.label}
                      className={`flex justify-between items-center py-3 ${
                        i < arr.length - 1 ? "border-b border-[#1f2835]/60" : ""
                      }`}
                    >
                      <span className={`text-xs ${tw.textTertiary}`}>{item.label}</span>
                      <span className={`text-sm ${tw.textPrimary} font-medium tabular-nums`}>
                        {item.value}
                      </span>
                    </div>
                  ))
              : null}
        </div>
      </div>
    </div>
  );
}
