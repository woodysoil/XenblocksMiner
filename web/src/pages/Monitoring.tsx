import { useMemo, useState, useEffect, useRef, useCallback } from "react";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "../api";
import { useDashboard } from "../context/DashboardContext";
import { tw, colors, chartTheme } from "../design/tokens";
import { StatusBadge, MetricCard, ChartCard, Skeleton } from "../design";
import Pagination from "../components/Pagination";
import { formatHashrate, timeAgo } from "../utils/format";
import type { Worker, Block, HashratePoint } from "../types";

// ── Helpers ──────────────────────────────────────────────────

function lastSeenColor(sec: number): string {
  if (sec < 300) return colors.success.DEFAULT;
  if (sec < 1800) return colors.warning.DEFAULT;
  return colors.danger.DEFAULT;
}

function gpuLabel(w: Worker): string {
  if (!w.gpus?.length) return `${w.gpu_count} GPU`;

  const counts = new Map<string, number>();
  for (const g of w.gpus) {
    const short = (g.name || "GPU").replace(/^NVIDIA\s+/, "").replace(/GeForce\s+/, "");
    counts.set(short, (counts.get(short) || 0) + 1);
  }

  if (counts.size === 1) {
    const [name, count] = [...counts.entries()][0];
    return count > 1 ? `${count}x ${name}` : name;
  } else if (counts.size === 2 && w.gpu_count <= 4) {
    return [...counts.entries()].map(([n, c]) => c > 1 ? `${c}x ${n}` : n).join(" + ");
  } else {
    return `${w.gpu_count}x (mixed)`;
  }
}

function gpuUtilization(w: Worker): string | null {
  if (!w.gpu_count) return null;
  const pct = Math.round((w.active_gpus / w.gpu_count) * 100);
  return `${w.active_gpus}/${w.gpu_count} (${pct}%)`;
}

function compactNumber(v: number): string {
  if (v >= 1e6) return (v / 1e6).toFixed(1) + "M";
  if (v >= 1e3) return (v / 1e3).toFixed(1) + "K";
  return v.toLocaleString();
}

type SortField = "name" | "gpu" | "hashrate" | "status" | "last_seen";
type SortDir = "asc" | "desc";
type StatusFilter = "all" | "online" | "offline";
type TimeRange = 1 | 6 | 24;

const TIME_RANGES: { label: string; hours: TimeRange }[] = [
  { label: "1h", hours: 1 },
  { label: "6h", hours: 6 },
  { label: "24h", hours: 24 },
];

const PAGE_SIZE = 20;

// ── Main Component ───────────────────────────────────────────

export default function Monitoring() {
  const { workers: wsWorkers, stats, recentBlocks, connected } = useDashboard();
  const queryClient = useQueryClient();

  // Table state
  const [sortField, setSortField] = useState<SortField>("hashrate");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [page, setPage] = useState(1);

  // Chart state
  const [chartRange, setChartRange] = useState<TimeRange>(1);

  // Data freshness
  const [lastUpdate, setLastUpdate] = useState(Date.now());
  const [freshness, setFreshness] = useState("just now");

  // Previous stats for delta
  const prevStats = useRef(stats);
  const [deltas, setDeltas] = useState<Record<string, string | undefined>>({});

  useEffect(() => {
    const prev = prevStats.current;
    if (prev.total_hashrate > 0 && stats.total_hashrate > 0 && prev.total_hashrate !== stats.total_hashrate) {
      const hrDelta = ((stats.total_hashrate - prev.total_hashrate) / prev.total_hashrate) * 100;
      const blkDelta = stats.blocks_last_hour - prev.blocks_last_hour;
      setDeltas({
        hashrate: `${hrDelta >= 0 ? "+" : ""}${hrDelta.toFixed(1)}%`,
        blocks_hr: blkDelta !== 0 ? `${blkDelta >= 0 ? "+" : ""}${blkDelta}` : undefined,
      });
    }
    prevStats.current = stats;
  }, [stats]);

  // Update freshness ticker
  useEffect(() => {
    if (wsWorkers.length > 0) setLastUpdate(Date.now());
  }, [wsWorkers, recentBlocks]);

  useEffect(() => {
    const tick = () => {
      const sec = Math.floor((Date.now() - lastUpdate) / 1000);
      if (sec < 5) setFreshness("just now");
      else if (sec < 60) setFreshness(`${sec}s ago`);
      else setFreshness(`${Math.floor(sec / 60)}m ago`);
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [lastUpdate]);

  // Chart data
  const bucketSize = chartRange <= 1 ? 30 : chartRange <= 6 ? 120 : 300;
  const { data: chartData = [] } = useQuery({
    queryKey: ["monitoring", "chart", chartRange],
    queryFn: async () => {
      const pts = await apiFetch<HashratePoint[]>(`/api/monitoring/hashrate-history?hours=${chartRange}`);
      const buckets = new Map<number, number>();
      for (const p of pts) {
        const key = Math.floor(p.timestamp / bucketSize) * bucketSize;
        buckets.set(key, (buckets.get(key) || 0) + p.hashrate);
      }
      return [...buckets.entries()]
        .sort((a, b) => a[0] - b[0])
        .map(([ts, hr]) => ({ ts, hashrate: hr }));
    },
    refetchInterval: 60_000,
  });

  // Fleet data — server sorts by hashrate only; client handles other sort fields
  const serverSortDir = sortField === "hashrate" ? sortDir : "desc";
  const { data: fleetData, isLoading: loading } = useQuery({
    queryKey: ["monitoring", "fleet", page, serverSortDir, statusFilter],
    queryFn: () => {
      const params = new URLSearchParams({
        page: String(page),
        limit: String(PAGE_SIZE),
        sort: serverSortDir,
      });
      if (statusFilter !== "all") params.set("status", statusFilter);
      return apiFetch<{ items: Worker[]; total_pages: number }>(`/api/monitoring/fleet?${params}`);
    },
  });
  const totalPages = fleetData?.total_pages ?? 1;

  // Client-side sort for non-hashrate fields
  const fleetWorkers = useMemo(() => {
    const items = [...(fleetData?.items ?? [])];
    if (sortField === "hashrate") return items;

    const dir = sortDir === "asc" ? 1 : -1;
    items.sort((a, b) => {
      switch (sortField) {
        case "name":
          return dir * a.worker_id.localeCompare(b.worker_id);
        case "gpu":
          return dir * (a.gpu_count - b.gpu_count);
        case "status": {
          const s = (w: Worker) => w.online ? 1 : 0;
          return dir * (s(a) - s(b));
        }
        case "last_seen":
          return dir * (a.last_heartbeat - b.last_heartbeat);
        default:
          return 0;
      }
    });
    return items;
  }, [fleetData?.items, sortField, sortDir]);

  // WebSocket real-time updates
  useEffect(() => {
    if (wsWorkers.length === 0 || !fleetData?.items?.length) return;
    const wsMap = new Map(wsWorkers.map((w) => [w.worker_id, w]));
    queryClient.setQueryData(
      ["monitoring", "fleet", page, serverSortDir, statusFilter],
      (old: typeof fleetData) => {
        if (!old) return old;
        return {
          ...old,
          items: old.items.map((w) => {
            const updated = wsMap.get(w.worker_id);
            if (!updated) return w;
            return {
              ...w,
              hashrate: updated.hashrate,
              active_gpus: updated.active_gpus,
              last_heartbeat: updated.last_heartbeat,
              online: updated.online,
            };
          }),
        };
      },
    );
  }, [wsWorkers, fleetData, page, serverSortDir, statusFilter, queryClient]);

  // Reset page on filter/sort change
  useEffect(() => { setPage(1); }, [sortDir, sortField, statusFilter]);

  const maxHashrate = useMemo(
    () => Math.max(...fleetWorkers.map((w) => w.hashrate), 1),
    [fleetWorkers],
  );

  const formatY = (v: number) => {
    if (v >= 1e6) return (v / 1e6).toFixed(1) + "M";
    if (v >= 1e3) return (v / 1e3).toFixed(1) + "K";
    return String(Math.round(v));
  };

  const toggleSort = useCallback((field: SortField) => {
    setSortField((prev) => {
      if (prev === field) {
        setSortDir((d) => (d === "desc" ? "asc" : "desc"));
      } else {
        setSortDir("desc");
      }
      return field;
    });
  }, []);

  const formatChartTime = (ts: number) => {
    const d = new Date(ts * 1000);
    if (chartRange <= 1) return d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
    if (chartRange <= 6) return d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  };

  // ── Render ──────────────────────────────────────────────────

  return (
    <div className="space-y-6">
      {/* Live indicator + freshness */}
      <div className="flex items-center gap-3">
        <span className="inline-flex items-center gap-1.5 text-xs">
          <span
            className="w-2 h-2 rounded-full"
            style={{
              backgroundColor: connected ? colors.success.DEFAULT : colors.danger.DEFAULT,
              boxShadow: connected ? `0 0 6px ${colors.success.DEFAULT}80` : undefined,
              animation: connected ? "pulse 2s infinite" : undefined,
            }}
          />
          <span style={{ color: connected ? colors.success.DEFAULT : colors.danger.DEFAULT }}>
            {connected ? "Live" : "Disconnected"}
          </span>
        </span>
        <span className={`text-xs ${tw.textTertiary}`}>Updated {freshness}</span>
      </div>

      {/* Platform Stats */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <MetricCard
          label="Workers"
          value={`${stats.online}/${stats.online + stats.offline}`}
          variant="success"
        />
        <MetricCard
          label="Total GPUs"
          value={compactNumber(stats.total_gpus)}
          variant="accent"
        />
        <MetricCard
          label="Active GPUs"
          value={compactNumber(stats.active_gpus)}
          variant="info"
        />
        <MetricCard
          label="Hashrate"
          value={formatHashrate(stats.total_hashrate)}
          delta={deltas.hashrate}
          variant="accent"
        />
        <MetricCard
          label="Total Blocks"
          value={compactNumber(stats.total_blocks)}
          variant="warning"
        />
        <MetricCard
          label="Blocks/hr"
          value={compactNumber(stats.blocks_last_hour)}
          delta={deltas.blocks_hr}
          variant="success"
        />
      </div>

      {/* Main grid */}
      <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">
        {/* Worker Fleet */}
        <div className={`xl:col-span-3 ${tw.card} overflow-hidden flex flex-col`}>
          <div className="px-4 pt-4 pb-2 flex items-center justify-between gap-3">
            <h3 className={`text-sm font-semibold ${tw.textPrimary}`}>Worker Fleet</h3>
            <div className="flex items-center gap-1">
              {(["all", "online", "offline"] as StatusFilter[]).map((f) => (
                <button
                  key={f}
                  onClick={() => setStatusFilter(f)}
                  className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                    statusFilter === f
                      ? `bg-[${colors.accent.muted}] text-[${colors.accent.DEFAULT}]`
                      : `${tw.textTertiary} hover:${tw.textSecondary}`
                  }`}
                  style={
                    statusFilter === f
                      ? { backgroundColor: colors.accent.muted, color: colors.accent.DEFAULT }
                      : undefined
                  }
                >
                  {f.charAt(0).toUpperCase() + f.slice(1)}
                </button>
              ))}
            </div>
          </div>
          <div className="overflow-x-auto flex-1">
            <table className="w-full text-sm min-w-[800px]">
              <thead>
                <tr className={`${tw.surface2} border-b border-[${colors.border.default}]`}>
                  <SortHeader field="name" label="Worker" current={sortField} dir={sortDir} onSort={toggleSort} />
                  <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Address</th>
                  <SortHeader field="gpu" label="GPU" current={sortField} dir={sortDir} onSort={toggleSort} />
                  <SortHeader field="hashrate" label="Hashrate" current={sortField} dir={sortDir} onSort={toggleSort} />
                  <SortHeader field="status" label="Status" current={sortField} dir={sortDir} onSort={toggleSort} />
                  <th className={`${tw.tableHeader} px-4 py-3 text-left`}>GPU Util</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Blocks</th>
                  <SortHeader field="last_seen" label="Last Seen" current={sortField} dir={sortDir} onSort={toggleSort} />
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  Array.from({ length: 5 }).map((_, i) => (
                    <tr key={i} className={`border-b border-[${colors.bg.surface3}]`}>
                      {Array.from({ length: 8 }).map((_, j) => (
                        <td key={j} className="px-4 py-3"><Skeleton className="h-4 w-20" /></td>
                      ))}
                    </tr>
                  ))
                ) : fleetWorkers.length === 0 ? (
                  <tr>
                    <td colSpan={8} className={`py-12 text-center text-sm ${tw.textSecondary}`}>
                      No workers connected
                    </td>
                  </tr>
                ) : (
                  fleetWorkers.map((w) => {
                    const pct = w.online ? (w.hashrate / maxHashrate) * 100 : 0;
                    const status: "online" | "offline" | "leased" = !w.online
                      ? "offline"
                      : w.state === "LEASED"
                        ? "leased"
                        : "online";
                    const seen = timeAgo(w.last_heartbeat);
                    const isOffline = !w.online;

                    return (
                      <tr
                        key={w.worker_id}
                        className={tw.tableRow}
                        style={isOffline ? { backgroundColor: colors.danger.muted } : undefined}
                      >
                        <td className={`${tw.tableCell} font-mono text-xs`}>
                          <span className="block truncate max-w-[120px]">{w.worker_id}</span>
                        </td>
                        <td className={`${tw.tableCell} font-mono text-xs`} style={{ color: colors.text.tertiary }}>
                          {w.eth_address
                            ? `${w.eth_address.slice(0, 5)}...${w.eth_address.slice(-4)}`
                            : "\u2014"}
                        </td>
                        <td className={tw.tableCell}>
                          <span
                            className="px-2 py-0.5 rounded text-xs font-mono"
                            style={{ backgroundColor: colors.bg.surface3, color: colors.text.secondary }}
                          >
                            {gpuLabel(w)}
                          </span>
                        </td>
                        <td className={tw.tableCell}>
                          <div className="flex items-center gap-2">
                            <span
                              className="font-mono text-xs tabular-nums w-20"
                              style={{ color: colors.accent.DEFAULT }}
                            >
                              {w.online ? formatHashrate(w.hashrate) : "\u2014"}
                            </span>
                            <div
                              className="flex-1 h-2 rounded-full overflow-hidden max-w-[120px]"
                              style={{ backgroundColor: colors.accent.muted }}
                            >
                              <div
                                className="h-full rounded-full"
                                style={{
                                  width: `${pct}%`,
                                  backgroundColor: "rgba(34,209,238,0.5)",
                                  boxShadow: `0 0 6px ${colors.accent.glow}`,
                                }}
                              />
                            </div>
                          </div>
                        </td>
                        <td className={tw.tableCell}>
                          <StatusBadge status={status} size="sm" />
                        </td>
                        <td className={`${tw.tableCell} font-mono text-xs`} style={{ color: colors.text.secondary }}>
                          {gpuUtilization(w) || "\u2014"}
                        </td>
                        <td className={`${tw.tableCell} tabular-nums`}>{w.self_blocks_found}</td>
                        <td className={`${tw.tableCell} text-xs tabular-nums`} style={{ color: lastSeenColor(seen.sec) }}>
                          {seen.text}
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
          <div className="px-4 pb-4">
            <Pagination currentPage={page} totalPages={totalPages} onPageChange={setPage} />
          </div>
        </div>

        {/* Right panels */}
        <div className="xl:col-span-2 space-y-6">
          {/* Hashrate chart */}
          <ChartCard
            title="Fleet Hashrate"
            action={
              <div className="flex items-center gap-1">
                {TIME_RANGES.map((r) => (
                  <button
                    key={r.hours}
                    onClick={() => setChartRange(r.hours)}
                    className="px-2 py-0.5 rounded text-xs font-medium transition-colors"
                    style={
                      chartRange === r.hours
                        ? { backgroundColor: colors.accent.muted, color: colors.accent.DEFAULT }
                        : { color: colors.text.tertiary }
                    }
                  >
                    {r.label}
                  </button>
                ))}
              </div>
            }
          >
            {chartData.length === 0 ? (
              <div className="h-56 flex items-center justify-center">
                <span className={`text-sm ${tw.textSecondary}`}>No hashrate data yet</span>
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={224}>
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="hrGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={chartTheme.areaGradient.start} />
                      <stop offset="100%" stopColor={chartTheme.areaGradient.end} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke={chartTheme.grid.stroke} strokeDasharray={chartTheme.grid.strokeDasharray} />
                  <XAxis
                    dataKey="ts"
                    tickFormatter={formatChartTime}
                    tick={{ fill: chartTheme.axis.fill, fontSize: chartTheme.axis.fontSize }}
                    stroke={chartTheme.axis.stroke}
                  />
                  <YAxis
                    tickFormatter={formatY}
                    tick={{ fill: chartTheme.axis.fill, fontSize: chartTheme.axis.fontSize }}
                    stroke={chartTheme.axis.stroke}
                    width={50}
                  />
                  <Tooltip
                    contentStyle={chartTheme.tooltip.contentStyle}
                    labelStyle={chartTheme.tooltip.labelStyle}
                    itemStyle={chartTheme.tooltip.itemStyle}
                    cursor={{ stroke: chartTheme.cursor.stroke }}
                    labelFormatter={(ts: number) =>
                      new Date(ts * 1000).toLocaleString("en-US", {
                        month: "short", day: "numeric",
                        hour: "2-digit", minute: "2-digit", second: "2-digit",
                      })
                    }
                    formatter={(v: number) => [formatHashrate(v), "Hashrate"]}
                  />
                  <Area
                    type="monotone"
                    dataKey="hashrate"
                    stroke={colors.accent.DEFAULT}
                    fill="url(#hrGrad)"
                    strokeWidth={2}
                  />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </ChartCard>

          {/* Block feed */}
          <div className={tw.card}>
            <div className="flex items-center gap-2 px-5 pt-4 pb-3">
              <h3 className={`text-sm font-semibold ${tw.textPrimary}`}>Live Blocks</h3>
              <span className={tw.badgeAccent}>{recentBlocks.length}</span>
            </div>
            <div className="px-5 pb-4 max-h-[360px] overflow-y-auto">
              {recentBlocks.length === 0 ? (
                <div className="py-8 text-center">
                  <span className={`text-sm ${tw.textSecondary}`}>No blocks found yet</span>
                </div>
              ) : (
                recentBlocks.slice(0, 15).map((b, i) => (
                  <BlockRow key={`${b.hash}-${i}`} block={b} isNew={i === 0} />
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────

function SortHeader({
  field, label, current, dir, onSort,
}: {
  field: SortField;
  label: string;
  current: SortField;
  dir: SortDir;
  onSort: (f: SortField) => void;
}) {
  const active = current === field;
  return (
    <th
      className={`${tw.tableHeader} px-4 py-3 text-left cursor-pointer select-none transition-colors`}
      style={active ? { color: colors.text.secondary } : undefined}
      onClick={() => onSort(field)}
      aria-sort={active ? (dir === "asc" ? "ascending" : "descending") : undefined}
      aria-label={`Sort by ${label}`}
    >
      {label}{" "}
      <span className="text-[10px] ml-0.5" style={{ opacity: active ? 1 : 0.3 }}>
        {active ? (dir === "asc" ? "\u25B2" : "\u25BC") : "\u25BC"}
      </span>
    </th>
  );
}

function BlockRow({ block: b, isNew }: { block: Block; isNew?: boolean }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isNew || !ref.current) return;
    const el = ref.current;
    el.style.opacity = "0";
    el.style.transform = "translateY(-12px)";
    requestAnimationFrame(() => {
      el.style.transition = "opacity 0.4s ease-out, transform 0.4s ease-out";
      el.style.opacity = "1";
      el.style.transform = "translateY(0)";
    });
  }, [isNew]);

  return (
    <div
      ref={ref}
      className={`py-2.5 border-b text-xs ${
        isNew ? `animate-[blockFlash_1s_ease-out]` : ""
      }`}
      style={{ borderBottomColor: colors.bg.surface3 }}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono truncate" style={{ color: colors.text.secondary }}>
          {b.hash.slice(0, 16)}{"\u2026"}
        </span>
        <div className="flex items-center gap-2 shrink-0">
          <span className="font-mono" style={{ color: colors.text.secondary }}>{b.worker_id}</span>
          {b.lease_id ? (
            <span className={tw.badgeInfo}>leased</span>
          ) : (
            <span className={tw.badgeSuccess}>self</span>
          )}
        </div>
      </div>
      <div className="flex items-center gap-3 mt-1" style={{ color: colors.text.tertiary }}>
        {b.hashrate && (
          <span className="font-mono tabular-nums">
            {formatHashrate(typeof b.hashrate === "string" ? parseFloat(b.hashrate) : b.hashrate)}
          </span>
        )}
        <span className="tabular-nums">
          {new Date(b.timestamp > 1e12 ? b.timestamp : b.timestamp * 1000).toLocaleTimeString("en-US", {
            hour: "2-digit", minute: "2-digit", second: "2-digit",
          })}
        </span>
        <span className="tabular-nums" style={{ minWidth: "3.5rem", textAlign: "right" }}>
          {timeAgo(b.timestamp).text}
        </span>
      </div>
    </div>
  );
}
