import { useMemo, useState, useEffect, useCallback } from "react";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import { useDashboard } from "../context/DashboardContext";
import { useWallet } from "../context/WalletContext";
import { tw, colors, chartTheme } from "../design/tokens";
import { Pill, StatusBadge, ChartCard } from "../design";
import Pagination from "../components/Pagination";
import type { Worker, Block, HashratePoint } from "../types";

function timeAgo(ts: number): string {
  const sec = Math.floor(Date.now() / 1000 - ts);
  if (sec < 60) return `${sec}s ago`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
  return `${Math.floor(sec / 3600)}h ago`;
}

function formatHashrate(h: number): string {
  if (h >= 1e9) return (h / 1e9).toFixed(2) + " GH/s";
  if (h >= 1e6) return (h / 1e6).toFixed(2) + " MH/s";
  if (h >= 1e3) return (h / 1e3).toFixed(2) + " KH/s";
  return h.toFixed(1) + " H/s";
}

function gpuLabel(w: Worker): string {
  if (!w.gpus?.length) return `${w.gpu_count} GPU`;

  // Aggregate GPU types
  const counts = new Map<string, number>();
  for (const g of w.gpus) {
    const short = (g.name || "GPU").replace(/^NVIDIA\s+/, "").replace(/GeForce\s+/, "");
    counts.set(short, (counts.get(short) || 0) + 1);
  }

  if (counts.size === 1) {
    // All same type
    const [name, count] = [...counts.entries()][0];
    return count > 1 ? `${count}x ${name}` : name;
  } else if (counts.size === 2 && w.gpu_count <= 4) {
    // 2 different types, show both
    return [...counts.entries()].map(([n, c]) => c > 1 ? `${c}x ${n}` : n).join(" + ");
  } else {
    // Many mixed types
    return `${w.gpu_count}x (mixed)`;
  }
}

type SortDir = "asc" | "desc";

const PAGE_SIZE = 20;

export default function Monitoring() {
  const { workers: wsWorkers, stats, recentBlocks } = useDashboard();
  const { address } = useWallet();
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [myOnly, setMyOnly] = useState(false);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [fleetWorkers, setFleetWorkers] = useState<Worker[]>([]);
  const [loading, setLoading] = useState(false);
  const [chartData, setChartData] = useState<{ time: string; hashrate: number }[]>([]);

  const fetchChart = useCallback(() => {
    fetch("/api/monitoring/hashrate-history?hours=1")
      .then((r) => r.json())
      .then((pts: HashratePoint[]) => {
        const buckets = new Map<number, number>();
        for (const p of pts) {
          const key = Math.floor(p.timestamp / 30) * 30;
          buckets.set(key, (buckets.get(key) || 0) + p.hashrate);
        }
        setChartData(
          [...buckets.entries()]
            .sort((a, b) => a[0] - b[0])
            .map(([ts, hr]) => ({
              time: new Date(ts * 1000).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" }),
              hashrate: hr,
            })),
        );
      })
      .catch(() => {});
  }, []);

  const fetchFleet = useCallback(() => {
    setLoading(true);
    const params = new URLSearchParams({
      page: String(page),
      limit: String(PAGE_SIZE),
      sort: sortDir,
    });
    if (myOnly && address) {
      params.set("eth_address", address);
    }
    fetch(`/api/monitoring/fleet?${params}`)
      .then((r) => r.json())
      .then((res: { items: Worker[]; total_pages: number }) => {
        setFleetWorkers(res.items || []);
        setTotalPages(res.total_pages || 1);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [page, sortDir, myOnly, address]);

  useEffect(() => {
    fetchChart();
    const id = setInterval(fetchChart, 60000);
    return () => clearInterval(id);
  }, [fetchChart]);

  useEffect(() => {
    fetchFleet();
  }, [fetchFleet]);

  // Apply WebSocket real-time updates to current page workers
  useEffect(() => {
    if (wsWorkers.length === 0 || fleetWorkers.length === 0) return;
    const wsMap = new Map(wsWorkers.map((w) => [w.worker_id, w]));
    setFleetWorkers((prev) =>
      prev.map((w) => {
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
    );
  }, [wsWorkers]);

  // Reset page when filters change
  useEffect(() => {
    setPage(1);
  }, [sortDir, myOnly]);

  const maxHashrate = useMemo(
    () => Math.max(...fleetWorkers.map((w) => w.hashrate), 1),
    [fleetWorkers],
  );

  const formatY = (v: number) => {
    if (v >= 1e6) return (v / 1e6).toFixed(1) + "M";
    if (v >= 1e3) return (v / 1e3).toFixed(1) + "K";
    return String(Math.round(v));
  };

  const gpuSummary = useMemo(() => {
    const counts = new Map<string, number>();
    for (const w of fleetWorkers) {
      if (!w.gpus?.length) continue;
      for (const g of w.gpus) {
        const short = (g.name || "GPU").replace(/^NVIDIA\s+/, "").replace(/GeForce\s+/, "");
        counts.set(short, (counts.get(short) || 0) + 1);
      }
    }
    return [...counts.entries()].map(([name, n]) => `${n}x ${name}`).join(", ") || `${stats.active_gpus} GPU`;
  }, [fleetWorkers, stats.active_gpus]);

  return (
    <div className="space-y-6">
      {/* Stat pills */}
      <div className="flex flex-wrap items-center gap-2">
        <Pill label="Online" value={stats.online} color="success" />
        <Pill label="Offline" value={stats.offline} color="danger" />
        <Pill label="Hashrate" value={formatHashrate(stats.total_hashrate)} color="accent" />
        <Pill label="GPUs" value={gpuSummary} />
        <Pill label="Blocks/hr" value={stats.blocks_last_hour} />
        {address && (
          <button
            onClick={() => setMyOnly((v) => !v)}
            className={`ml-auto px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              myOnly
                ? "bg-[#22d1ee]/20 border border-[#22d1ee]/50 text-[#22d1ee]"
                : "bg-[#1f2835] border border-[#2a3441] text-[#848e9c] hover:text-[#eaecef]"
            }`}
          >
            My Miners
          </button>
        )}
      </div>

      {/* Main grid */}
      <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">
        {/* Worker Fleet */}
        <div className={`xl:col-span-3 ${tw.card} overflow-hidden flex flex-col`}>
          <div className="px-4 pt-4 pb-2">
            <h3 className={`text-sm font-semibold ${tw.textPrimary}`}>Worker Fleet</h3>
          </div>
          <div className="overflow-x-auto flex-1">
            <table className="w-full text-sm">
              <thead>
                <tr className={`${tw.surface2} border-b border-[#2a3441]`}>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Worker</th>
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
                    <td colSpan={6} className="py-12 text-center text-[#848e9c] text-sm">
                      Loading...
                    </td>
                  </tr>
                ) : fleetWorkers.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="py-12 text-center text-[#848e9c] text-sm">
                      {myOnly ? "No miners match your wallet address" : "No workers connected"}
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
                    return (
                      <tr key={w.worker_id} className={tw.tableRow}>
                        <td className={`${tw.tableCell} font-mono text-xs`}>
                          {w.worker_id}
                        </td>
                        <td className={tw.tableCell}>
                          <span className="bg-[#1f2835] px-2 py-0.5 rounded text-xs font-mono text-[#848e9c]">
                            {gpuLabel(w)}
                          </span>
                        </td>
                        <td className={tw.tableCell}>
                          <div className="flex items-center gap-2">
                            <span className="font-mono text-xs text-[#22d1ee] w-20">
                              {w.online ? formatHashrate(w.hashrate) : "\u2014"}
                            </span>
                            <div className="flex-1 h-1.5 bg-[rgba(34,209,238,0.1)] rounded-full overflow-hidden max-w-[120px]">
                              <div
                                className="h-full rounded-full bg-[rgba(34,209,238,0.5)]"
                                style={{ width: `${pct}%` }}
                              />
                            </div>
                          </div>
                        </td>
                        <td className={tw.tableCell}>
                          <StatusBadge status={status} size="sm" />
                        </td>
                        <td className={tw.tableCell}>{w.self_blocks_found}</td>
                        <td className={`${tw.tableCell} text-[#5e6673] text-xs`}>
                          {timeAgo(w.last_heartbeat)}
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
      <span className="font-mono truncate text-[#eaecef]">{b.hash.slice(0, 16)}\u2026</span>
      <div className="flex items-center gap-2 shrink-0">
        <span className="font-mono text-[#848e9c]">{b.worker_id}</span>
        {b.lease_id ? (
          <span className={tw.badgeInfo}>leased</span>
        ) : (
          <span className="text-xs px-2 py-0.5 rounded bg-[#1f2835] text-[#848e9c]">self</span>
        )}
        <span className="text-[#5e6673] w-14 text-right">{timeAgo(b.timestamp)}</span>
      </div>
    </div>
  );
}
