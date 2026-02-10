import { useEffect, useState, useCallback } from "react";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import { tw, colors, chartTheme } from "../design/tokens";
import MetricCard from "../design/MetricCard";
import StatusBadge from "../design/StatusBadge";
import GpuBadge from "../design/GpuBadge";
import HashText from "../design/HashText";
import { ChartCard } from "../design";
import EmptyState from "../design/EmptyState";
import { useWallet } from "../context/WalletContext";
import { apiFetch } from "../api";
import type { WalletSnapshot, WalletAchievements } from "../types";

interface MyWorker {
  worker_id: string;
  gpu_name: string;
  memory_gb: number;
  hashrate: number;
  state: string;
  price_per_min: number;
  self_blocks_found: number;
  online: boolean;
  total_online_sec: number;
}

interface Dashboard {
  total_earned: number;
  active_leases: number;
  total_blocks_mined: number;
  avg_hashrate: number;
  worker_count: number;
}

function formatHashrate(h: number): string {
  if (h >= 1e9) return (h / 1e9).toFixed(2) + " GH/s";
  if (h >= 1e6) return (h / 1e6).toFixed(2) + " MH/s";
  if (h >= 1e3) return (h / 1e3).toFixed(2) + " KH/s";
  return h.toFixed(1) + " H/s";
}

function formatUptime(sec: number): string {
  const d = Math.floor(sec / 86400);
  const h = Math.floor((sec % 86400) / 3600);
  if (d > 0) return `${d}d ${h}h`;
  if (h > 0) return `${h}h`;
  return `${Math.floor(sec / 60)}m`;
}

const stateToStatus: Record<string, "idle" | "available" | "leased"> = {
  SELF_MINING: "idle",
  AVAILABLE: "available",
  LEASED: "leased",
};

type Period = "24h" | "7d" | "30d";

export default function Provider() {
  const { address, connect } = useWallet();
  const [workers, setWorkers] = useState<MyWorker[]>([]);
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);
  const [achievements, setAchievements] = useState<WalletAchievements | null>(null);
  const [history, setHistory] = useState<WalletSnapshot[]>([]);
  const [period, setPeriod] = useState<Period>("24h");

  const fetchData = useCallback(() => {
    if (!address) return;
    apiFetch<{ items: MyWorker[] }>("/api/provider/workers")
      .then((d) => setWorkers(d.items || []))
      .catch(() => {});
    apiFetch<Dashboard>("/api/provider/dashboard")
      .then((d) => setDashboard(d))
      .catch(() => {});
  }, [address]);

  const fetchHistory = useCallback(() => {
    if (!address) return;
    apiFetch<{ data: WalletSnapshot[] }>(`/api/wallet/history?period=${period}`)
      .then((d) => setHistory(d.data || []))
      .catch(() => setHistory([]));
    apiFetch<WalletAchievements>("/api/wallet/achievements")
      .then((d) => setAchievements(d))
      .catch(() => {});
  }, [address, period]);

  useEffect(() => {
    fetchData();
    fetchHistory();
    const id = setInterval(fetchData, 10000);
    return () => clearInterval(id);
  }, [fetchData, fetchHistory]);

  if (!address) {
    return (
      <EmptyState
        icon={
          <svg width="32" height="32" viewBox="0 0 20 20" fill="none" stroke="#22d1ee" strokeWidth="1.5">
            <path d="M10 2l7 4v4l-7 4-7-4V6l7-4z" />
            <path d="M10 10v8" />
            <path d="M3 6l7 4 7-4" />
          </svg>
        }
        title="Connect your wallet to view your miners"
        description="Link your Ethereum wallet to manage workers, track earnings, and view mining history."
        action={
          <button
            onClick={connect}
            className="px-4 py-2 rounded-md bg-[#22d1ee]/10 border border-[#22d1ee]/30 text-sm font-medium text-[#22d1ee] hover:bg-[#22d1ee]/20 transition-colors"
          >
            Connect Wallet
          </button>
        }
      />
    );
  }

  const totalEarned = dashboard?.total_earned ?? 0;
  const totalBlocks = achievements?.total_blocks ?? dashboard?.total_blocks_mined ?? 0;
  const avgHashrate = dashboard?.avg_hashrate ?? 0;
  const peakHashrate = achievements?.peak_hashrate ?? 0;
  const miningDays = achievements?.mining_days ?? 0;

  const chartData = history.map((h) => ({
    time: new Date(h.timestamp * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    hashrate: h.hashrate,
    blocks: h.cumulative_blocks,
  }));

  const formatY = (v: number) => {
    if (v >= 1e6) return (v / 1e6).toFixed(1) + "M";
    if (v >= 1e3) return (v / 1e3).toFixed(1) + "K";
    return String(Math.round(v));
  };

  return (
    <div className="space-y-6">
      {/* Header with Share */}
      <div className="flex items-center justify-between">
        <h2 className={`text-xl font-semibold ${tw.textPrimary}`}>Provider Dashboard</h2>
        <button
          onClick={() => {
            const text = `Mining on XenBlocks!\n\n` +
              `Blocks: ${totalBlocks.toLocaleString()}\n` +
              `Peak Hashrate: ${formatHashrate(peakHashrate)}\n` +
              `Mining Days: ${miningDays}\n\n` +
              `#XenBlocks #Mining`;
            navigator.clipboard.writeText(text);
          }}
          className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-[#1f2835] border border-[#2a3441] text-sm text-[#848e9c] hover:text-[#eaecef] transition-colors"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M4 12v8a2 2 0 002 2h12a2 2 0 002-2v-8" />
            <polyline points="16 6 12 2 8 6" />
            <line x1="12" y1="2" x2="12" y2="15" />
          </svg>
          Share
        </button>
      </div>

      {/* Achievement Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
        <MetricCard label="Total Blocks" value={totalBlocks.toLocaleString()} variant="accent" />
        <MetricCard label="Peak Hashrate" value={formatHashrate(peakHashrate)} variant="info" />
        <MetricCard label="Total Earned" value={`${totalEarned.toFixed(2)} XNM`} variant="success" />
        <MetricCard label="Mining Days" value={miningDays.toFixed(0)} variant="warning" />
        <div
          className={`${tw.card} ${tw.cardHover} p-5 border-t-2`}
          style={{ borderTopColor: colors.text.tertiary }}
        >
          <span className={`text-xs ${tw.textTertiary} uppercase tracking-wider`}>Workers</span>
          <div className={`text-2xl font-bold ${tw.textPrimary} mt-1`}>
            {workers.filter(w => w.online).length}/{workers.length}
          </div>
        </div>
      </div>

      {/* History Chart */}
      <ChartCard
        title="Hashrate History"
        action={
          <div className="flex gap-1">
            {(["24h", "7d", "30d"] as Period[]).map((p) => (
              <button
                key={p}
                onClick={() => setPeriod(p)}
                className={`px-2 py-1 text-xs rounded ${
                  period === p
                    ? "bg-[#22d1ee]/20 text-[#22d1ee]"
                    : "text-[#848e9c] hover:text-[#eaecef]"
                }`}
              >
                {p}
              </button>
            ))}
          </div>
        }
      >
        {chartData.length === 0 ? (
          <div className="h-48 flex items-center justify-center">
            <span className={`text-sm ${tw.textSecondary}`}>
              {history.length === 0 ? "No history data yet — snapshots are taken hourly" : "Loading..."}
            </span>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={192}>
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

      {/* Current Hashrate Bar */}
      <div className={tw.card}>
        <div className="px-5 py-4">
          <div className="flex items-center justify-between mb-2">
            <span className={`text-sm font-medium ${tw.textPrimary}`}>Current Hashrate</span>
            <span className="text-lg font-bold" style={{ color: colors.accent.DEFAULT }}>
              {formatHashrate(avgHashrate)}
            </span>
          </div>
          <div className="h-2 bg-[#1f2835] rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{
                width: peakHashrate > 0 ? `${Math.min((avgHashrate / peakHashrate) * 100, 100)}%` : "0%",
                background: `linear-gradient(90deg, ${colors.accent.DEFAULT}, ${colors.info.DEFAULT})`,
              }}
            />
          </div>
          <div className="flex justify-between mt-1">
            <span className={`text-xs ${tw.textTertiary}`}>0</span>
            <span className={`text-xs ${tw.textTertiary}`}>Peak: {formatHashrate(peakHashrate)}</span>
          </div>
        </div>
      </div>

      {/* My Workers */}
      <div>
        <h3 className={`${tw.sectionTitle} mb-3`}>My Workers ({workers.length})</h3>
        {workers.length === 0 ? (
          <EmptyState title="No workers found" description="Make sure your miners are registered with this wallet address." />
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
            {workers.map((w) => (
              <div key={w.worker_id} className={`${tw.card} p-5`}>
                <div className="flex items-center justify-between mb-2">
                  <span className={`font-mono text-sm font-medium ${tw.textPrimary}`}>{w.worker_id}</span>
                  <StatusBadge status={stateToStatus[w.state] || "idle"} size="sm" />
                </div>
                <div className="mb-3">
                  <GpuBadge name={w.gpu_name || "GPU"} memory={w.memory_gb || 0} />
                </div>
                <div className="grid grid-cols-3 gap-3 text-sm">
                  <div>
                    <p className={`text-xs ${tw.textTertiary}`}>Hashrate</p>
                    <p className={`font-mono font-semibold ${tw.textPrimary}`}>{formatHashrate(w.hashrate)}</p>
                  </div>
                  <div>
                    <p className={`text-xs ${tw.textTertiary}`}>Blocks</p>
                    <p className={`font-mono ${tw.textPrimary}`}>{w.self_blocks_found}</p>
                  </div>
                  <div>
                    <p className={`text-xs ${tw.textTertiary}`}>Uptime</p>
                    <p className={tw.textPrimary}>{formatUptime(w.total_online_sec)}</p>
                  </div>
                </div>
                <div className="mt-3 pt-3 border-t border-[#1f2835] flex items-center justify-between">
                  <span className="text-xs font-mono" style={{ color: colors.warning.DEFAULT }}>
                    {w.price_per_min.toFixed(4)} <span className={tw.textTertiary}>/min</span>
                  </span>
                  <span className={`text-xs ${w.online ? "text-[#0ecb81]" : "text-[#f6465d]"}`}>
                    {w.online ? "Online" : "Offline"}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
