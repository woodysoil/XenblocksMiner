import { useMemo, useState, useEffect } from "react";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "../api";
import { useDashboard } from "../context/DashboardContext";
import { tw, colors, chartTheme } from "../design/tokens";
import { StatusBadge, ChartCard } from "../design";
import Pagination from "../components/Pagination";
import type { Worker, Block, HashratePoint } from "../types";

function timeAgo(ts: number): { text: string; sec: number } {
  const sec = Math.floor(Date.now() / 1000 - ts);
  let text: string;
  if (sec < 60) text = `${sec}s ago`;
  else if (sec < 3600) text = `${Math.floor(sec / 60)}m ago`;
  else text = `${Math.floor(sec / 3600)}h ago`;
  return { text, sec };
}

function lastSeenColor(sec: number): string {
  if (sec < 300) return "text-[#0ecb81]";   // <5m  green
  if (sec < 1800) return "text-[#f0b90b]";  // <30m yellow
  return "text-[#f6465d]";                   // >30m red
}

function formatHashrate(h: number): string {
  if (h >= 1e9) return (h / 1e9).toFixed(2) + " GH/s";
  if (h >= 1e6) return (h / 1e6).toFixed(2) + " MH/s";
  if (h >= 1e3) return (h / 1e3).toFixed(2) + " KH/s";
  return h.toFixed(1) + " H/s";
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

type SortDir = "asc" | "desc";

const PAGE_SIZE = 20;

export default function Monitoring() {
  const { workers: wsWorkers, stats, recentBlocks } = useDashboard();
  const queryClient = useQueryClient();
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [page, setPage] = useState(1);

  // Chart data (poll every 60s)
  const { data: chartData = [] } = useQuery({
    queryKey: ["monitoring", "chart"],
    queryFn: async () => {
      const pts = await apiFetch<HashratePoint[]>("/api/monitoring/hashrate-history?hours=1");
      const buckets = new Map<number, number>();
      for (const p of pts) {
        const key = Math.floor(p.timestamp / 30) * 30;
        buckets.set(key, (buckets.get(key) || 0) + p.hashrate);
      }
      return [...buckets.entries()]
        .sort((a, b) => a[0] - b[0])
        .map(([ts, hr]) => ({
          time: new Date(ts * 1000).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" }),
          hashrate: hr,
        }));
    },
    refetchInterval: 60_000,
  });

  // Fleet data (depends on page and sortDir)
  const { data: fleetData, isLoading: loading } = useQuery({
    queryKey: ["monitoring", "fleet", page, sortDir],
    queryFn: () => {
      const params = new URLSearchParams({ page: String(page), limit: String(PAGE_SIZE), sort: sortDir });
      return apiFetch<{ items: Worker[]; total_pages: number }>(`/api/monitoring/fleet?${params}`);
    },
  });
  const fleetWorkers = fleetData?.items ?? [];
  const totalPages = fleetData?.total_pages ?? 1;

  // Apply WebSocket real-time updates to current page workers
  useEffect(() => {
    if (wsWorkers.length === 0 || !fleetData?.items?.length) return;
    const wsMap = new Map(wsWorkers.map((w) => [w.worker_id, w]));
    queryClient.setQueryData(["monitoring", "fleet", page, sortDir], (old: typeof fleetData) => {
      if (!old) return old;
      return {
        ...old,
        items: old.items.map((w) => {
          const updated = wsMap.get(w.worker_id);
          if (!updated) return w;
          return { ...w, hashrate: updated.hashrate, active_gpus: updated.active_gpus, last_heartbeat: updated.last_heartbeat, online: updated.online };
        }),
      };
    });
  }, [wsWorkers, fleetData, page, sortDir, queryClient]);

  // Reset page when sort changes
  useEffect(() => {
    setPage(1);
  }, [sortDir]);

  const maxHashrate = useMemo(
    () => Math.max(...fleetWorkers.map((w) => w.hashrate), 1),
    [fleetWorkers],
  );

  const formatY = (v: number) => {
    if (v >= 1e6) return (v / 1e6).toFixed(1) + "M";
    if (v >= 1e3) return (v / 1e3).toFixed(1) + "K";
    return String(Math.round(v));
  };

  return (
    <div className="space-y-6">
      {/* Platform Stats */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <MiniStat label="Workers" value={`${stats.online}/${stats.online + stats.offline}`} sub="online" color={colors.success.DEFAULT} />
        <MiniStat label="Total GPUs" value={stats.total_gpus} color={colors.accent.DEFAULT} />
        <MiniStat label="Active GPUs" value={stats.active_gpus} color={colors.info.DEFAULT} />
        <MiniStat label="Hashrate" value={formatHashrate(stats.total_hashrate)} color={colors.accent.DEFAULT} />
        <MiniStat label="Total Blocks" value={stats.total_blocks} color={colors.warning.DEFAULT} />
        <MiniStat label="Blocks/hr" value={stats.blocks_last_hour} color={colors.success.DEFAULT} />
      </div>

      {/* Main grid */}
      <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">
        {/* Worker Fleet */}
        <div className={`xl:col-span-3 ${tw.card} overflow-hidden flex flex-col`}>
          <div className="px-4 pt-4 pb-2">
            <h3 className={`text-sm font-semibold ${tw.textPrimary}`}>Worker Fleet</h3>
          </div>
          <div className="overflow-x-auto flex-1">
            <table className="w-full text-sm min-w-[700px]">
              <thead>
                <tr className={`${tw.surface2} border-b border-[#2a3441]`}>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Worker</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Address</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left`}>GPU</th>
                  <th
                    className={`${tw.tableHeader} px-4 py-3 text-left cursor-pointer select-none hover:text-[#eaecef]`}
                    onClick={() => setSortDir((d) => (d === "desc" ? "asc" : "desc"))}
                  >
                    Hashrate{" "}
                    <span className="text-[10px] ml-0.5">{sortDir === "asc" ? "\u25B2" : "\u25BC"}</span>
                  </th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Status</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Blocks</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Last Seen</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr>
                    <td colSpan={7} className="py-12 text-center text-[#848e9c] text-sm">
                      Loading...
                    </td>
                  </tr>
                ) : fleetWorkers.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="py-12 text-center text-[#848e9c] text-sm">
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
                    return (
                      <tr key={w.worker_id} className={tw.tableRow}>
                        <td className={`${tw.tableCell} font-mono text-xs`}>
                          <span className="block truncate max-w-[120px]">{w.worker_id}</span>
                        </td>
                        <td className={`${tw.tableCell} font-mono text-xs text-[#6b7785]`}>
                          {w.eth_address
                            ? `${w.eth_address.slice(0, 5)}...${w.eth_address.slice(-4)}`
                            : "\u2014"}
                        </td>
                        <td className={tw.tableCell}>
                          <span className="bg-[#1f2835] px-2 py-0.5 rounded text-xs font-mono text-[#848e9c]">
                            {gpuLabel(w)}
                          </span>
                        </td>
                        <td className={tw.tableCell}>
                          <div className="flex items-center gap-2">
                            <span className="font-mono text-xs tabular-nums text-[#22d1ee] w-20">
                              {w.online ? formatHashrate(w.hashrate) : "\u2014"}
                            </span>
                            <div className="flex-1 h-2 bg-[rgba(34,209,238,0.08)] rounded-full overflow-hidden max-w-[120px]">
                              <div
                                className="h-full rounded-full bg-[rgba(34,209,238,0.5)] shadow-[0_0_6px_rgba(34,209,238,0.35)]"
                                style={{ width: `${pct}%` }}
                              />
                            </div>
                          </div>
                        </td>
                        <td className={tw.tableCell}>
                          <StatusBadge status={status} size="sm" />
                        </td>
                        <td className={`${tw.tableCell} tabular-nums`}>{w.self_blocks_found}</td>
                        <td className={`${tw.tableCell} text-xs tabular-nums ${lastSeenColor(seen.sec)}`}>
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
            <Pagination
              currentPage={page}
              totalPages={totalPages}
              onPageChange={setPage}
            />
          </div>
        </div>

        {/* Right panels */}
        <div className="xl:col-span-2 space-y-6">
          {/* Hashrate chart */}
          <ChartCard
            title="Fleet Hashrate"
            action={<span className={tw.badgeAccent}>1h</span>}
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
                      <stop offset="5%" stopColor="rgba(34,209,238,0.25)" />
                      <stop offset="95%" stopColor="rgba(34,209,238,0)" />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke={chartTheme.grid.stroke} strokeDasharray={chartTheme.grid.strokeDasharray} />
                  <XAxis
                    dataKey="time"
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

function BlockRow({ block: b, isNew }: { block: Block; isNew?: boolean }) {
  return (
    <div
      className={`flex items-center justify-between gap-2 py-2.5 border-b border-[#1f2835] text-xs ${
        isNew ? "animate-[blockFlash_1s_ease-out]" : ""
      }`}
    >
      <span className="font-mono truncate text-[#848e9c]">{b.hash.slice(0, 16)}{"\u2026"}</span>
      <div className="flex items-center gap-2 shrink-0">
        <span className="font-mono text-[#848e9c]">{b.worker_id}</span>
        {b.lease_id ? (
          <span className={tw.badgeInfo}>leased</span>
        ) : (
          <span className={tw.badgeSuccess}>self</span>
        )}
        <span className="text-[#5e6673] tabular-nums w-14 text-right">{timeAgo(b.timestamp).text}</span>
      </div>
    </div>
  );
}

function MiniStat({ label, value, sub, color }: { label: string; value: string | number; sub?: string; color?: string }) {
  return (
    <div
      className={`${tw.card} px-4 py-3`}
      style={color ? { borderLeftWidth: 2, borderLeftColor: color } : undefined}
    >
      <span className={`text-xs ${tw.textTertiary} uppercase tracking-wider`}>{label}</span>
      <div className="flex items-baseline gap-1.5 mt-0.5">
        <span className="text-lg font-bold tabular-nums" style={color ? { color } : undefined}>
          {typeof value === "number" ? value.toLocaleString() : value}
        </span>
        {sub && <span className={`text-xs ${tw.textTertiary}`}>{sub}</span>}
      </div>
    </div>
  );
}
