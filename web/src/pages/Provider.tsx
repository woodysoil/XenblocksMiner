import { useState, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import type { UTCTimestamp } from "lightweight-charts";
import { tw, colors } from "../design/tokens";
import MetricCard from "../design/MetricCard";
import StatusBadge from "../design/StatusBadge";
import GpuBadge from "../design/GpuBadge";
import { ChartCard } from "../design";
import LWChart from "../design/LWChart";
import EmptyState from "../design/EmptyState";
import ConfirmDialog from "../design/ConfirmDialog";
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

const stateToStatus: Record<string, "mining" | "available" | "leased"> = {
  SELF_MINING: "mining",
  AVAILABLE: "available",
  LEASED: "leased",
};

const stateToLabel: Record<string, string> = {
  SELF_MINING: "Mining",
  AVAILABLE: "Listed",
  LEASED: "Leased",
};

type FilterTab = "ALL" | "SELF_MINING" | "AVAILABLE" | "LEASED";
const filterTabs: { key: FilterTab; label: string }[] = [
  { key: "ALL", label: "All" },
  { key: "SELF_MINING", label: "Mining" },
  { key: "AVAILABLE", label: "Listed" },
  { key: "LEASED", label: "Leased" },
];

export default function Provider() {
  const { address, connect } = useWallet();
  const queryClient = useQueryClient();

  const [viewWindow, setViewWindow] = useState(7 * 86400);
  const [filter, setFilter] = useState<FilterTab>("ALL");
  const [confirmAction, setConfirmAction] = useState<{
    workerId: string;
    targetState: "AVAILABLE" | "SELF_MINING";
  } | null>(null);

  const { data: workersData } = useQuery({
    queryKey: ["provider", "workers", address],
    queryFn: () => apiFetch<{ items: MyWorker[] }>("/api/provider/workers"),
    refetchInterval: 10_000,
    enabled: !!address,
  });
  const workers = workersData?.items ?? [];

  const { data: dashboard } = useQuery({
    queryKey: ["provider", "dashboard", address],
    queryFn: () => apiFetch<Dashboard>("/api/provider/dashboard"),
    refetchInterval: 10_000,
    enabled: !!address,
  });

  const { data: historyData } = useQuery({
    queryKey: ["provider", "history", address],
    queryFn: () => apiFetch<{ data: WalletSnapshot[] }>("/api/wallet/history?period=30d"),
    enabled: !!address,
  });
  const history = historyData?.data ?? [];

  const { data: achievements } = useQuery({
    queryKey: ["provider", "achievements", address],
    queryFn: () => apiFetch<WalletAchievements>("/api/wallet/achievements"),
    enabled: !!address,
  });

  const commandMutation = useMutation({
    mutationFn: ({ workerId, targetState }: { workerId: string; targetState: string }) =>
      apiFetch(`/api/wallet/workers/${workerId}/command`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ command: "update_config", params: { state: targetState } }),
      }),
    onSuccess: () => {
      toast.success("Worker state updated");
      queryClient.invalidateQueries({ queryKey: ["provider", "workers"] });
    },
  });

  const filteredWorkers = useMemo(
    () => (filter === "ALL" ? workers : workers.filter((w) => w.state === filter)),
    [workers, filter],
  );

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

  const chartData = useMemo(
    () => history.map((h) => ({ time: h.timestamp as UTCTimestamp, value: h.hashrate })),
    [history],
  );

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
            navigator.clipboard.writeText(text).then(() => toast.success("Copied to clipboard!"));
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
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
        <MetricCard label="Total Blocks" value={totalBlocks.toLocaleString()} variant="accent" />
        <MetricCard label="Peak Hashrate" value={formatHashrate(peakHashrate)} variant="info" />
        <MetricCard label="Total Earned" value={`${totalEarned.toFixed(2)} XNM`} variant="success" />
        <MetricCard label="Mining Days" value={miningDays.toFixed(0)} variant="warning" />
        <div
          className={`${tw.card} ${tw.cardHover} p-5 border-t-2 border-t-[#5e6673]`}
        >
          <span className={`text-xs ${tw.textTertiary} uppercase tracking-wider`}>Workers</span>
          <div className={`text-2xl font-bold ${tw.textPrimary} mt-1 tabular-nums`}>
            {workers.filter(w => w.online).length}/{workers.length}
          </div>
        </div>
      </div>

      {/* History Chart */}
      <ChartCard
        title="Hashrate History"
        action={
          <div className="flex gap-1">
            {([
              { label: "24h", sec: 86400 },
              { label: "7d", sec: 7 * 86400 },
              { label: "30d", sec: 30 * 86400 },
              { label: "All", sec: 0 },
            ] as const).map((p) => (
              <button
                key={p.label}
                onClick={() => setViewWindow(p.sec)}
                className={`px-2 py-1 text-xs rounded ${
                  viewWindow === p.sec
                    ? "bg-[#22d1ee]/20 text-[#22d1ee]"
                    : "text-[#848e9c] hover:text-[#eaecef]"
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>
        }
      >
        {chartData.length === 0 ? (
          <div className="h-[260px] flex items-center justify-center">
            <span className={`text-sm ${tw.textSecondary}`}>
              No history data yet — snapshots are taken hourly
            </span>
          </div>
        ) : (
          <LWChart data={chartData} height={260} formatValue={formatHashrate} visibleWindow={viewWindow} />
        )}
      </ChartCard>

      {/* Current Hashrate Bar */}
      <div className={tw.card}>
        <div className="px-5 py-4">
          <div className="flex items-center justify-between mb-2">
            <span className={`text-sm font-medium ${tw.textPrimary}`}>Current Hashrate</span>
            <span className="text-lg font-bold tabular-nums" style={{ color: colors.accent.DEFAULT }}>
              {formatHashrate(avgHashrate)}
            </span>
          </div>
          <div className="h-2 bg-[#1f2835] rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{
                width: peakHashrate > 0 ? `${Math.min((avgHashrate / peakHashrate) * 100, 100)}%` : "0%",
                background: `linear-gradient(90deg, ${colors.accent.DEFAULT}, ${colors.info.DEFAULT})`,
                boxShadow: avgHashrate > 0 ? `0 0 8px ${colors.accent.glow}, 4px 0 12px ${colors.info.DEFAULT}40` : undefined,
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
        {/* Filter tabs */}
        <div className="flex flex-wrap gap-1 mb-4 border-b border-[#1f2835] pb-px">
          {filterTabs.map((t) => {
            const count = t.key === "ALL" ? workers.length : workers.filter((w) => w.state === t.key).length;
            const active = filter === t.key;
            return (
              <button
                key={t.key}
                onClick={() => setFilter(t.key)}
                className={`relative px-3 py-1.5 text-xs font-medium transition-colors ${
                  active
                    ? "text-[#22d1ee]"
                    : "text-[#848e9c] hover:text-[#eaecef]"
                }`}
              >
                {t.label}
                <span className={`ml-1.5 inline-flex items-center justify-center rounded-full px-1.5 py-0.5 text-[10px] leading-none tabular-nums ${
                  active
                    ? "bg-[rgba(34,209,238,0.15)] text-[#22d1ee]"
                    : "bg-[#1a2029] text-[#5e6673]"
                }`}>
                  {count}
                </span>
                {active && (
                  <span className="absolute bottom-0 left-1 right-1 h-0.5 rounded-full bg-[#22d1ee] shadow-[0_0_6px_rgba(34,209,238,0.4)]" />
                )}
              </button>
            );
          })}
        </div>
        {filteredWorkers.length === 0 ? (
          <EmptyState
            title={filter === "ALL" ? "No workers found" : `No ${filterTabs.find((t) => t.key === filter)?.label.toLowerCase()} workers`}
            description={filter === "ALL" ? "Make sure your miners are registered with this wallet address." : "Try a different filter."}
          />
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
            {filteredWorkers.map((w) => (
              <div key={w.worker_id} className={`${tw.card} ${tw.cardHover} p-5`}>
                <div className="flex items-center justify-between mb-2">
                  <span className={`font-mono text-sm font-medium ${tw.textPrimary} truncate max-w-[180px]`} title={w.worker_id}>{w.worker_id}</span>
                  <StatusBadge status={stateToStatus[w.state] || "idle"} label={stateToLabel[w.state]} size="sm" />
                </div>
                <div className="mb-3">
                  <GpuBadge name={w.gpu_name || "GPU"} memory={w.memory_gb || 0} />
                </div>
                <div className="grid grid-cols-3 gap-3 text-sm">
                  <div>
                    <p className={`text-xs ${tw.textTertiary}`}>Hashrate</p>
                    <p className={`font-mono font-semibold tabular-nums ${tw.textPrimary}`}>{formatHashrate(w.hashrate)}</p>
                  </div>
                  <div>
                    <p className={`text-xs ${tw.textTertiary}`}>Blocks</p>
                    <p className={`font-mono tabular-nums ${tw.textPrimary}`}>{w.self_blocks_found}</p>
                  </div>
                  <div>
                    <p className={`text-xs ${tw.textTertiary}`}>Uptime</p>
                    <p className={`tabular-nums ${tw.textPrimary}`}>{formatUptime(w.total_online_sec)}</p>
                  </div>
                </div>
                <div className="mt-3 pt-3 border-t border-[#1f2835] flex items-center justify-between">
                  <span className="text-xs font-mono tabular-nums" style={{ color: colors.warning.DEFAULT }}>
                    {w.price_per_min.toFixed(4)} <span className={tw.textTertiary}>/min</span>
                  </span>
                  <div className="flex items-center gap-3">
                    {w.state === "LEASED" ? (
                      <span className={`text-xs font-medium ${tw.textTertiary}`}>Leased</span>
                    ) : w.state === "AVAILABLE" ? (
                      <button
                        disabled={commandMutation.isPending && commandMutation.variables?.workerId === w.worker_id}
                        onClick={() => setConfirmAction({ workerId: w.worker_id, targetState: "SELF_MINING" })}
                        className="inline-flex items-center gap-1 px-2.5 py-1 text-xs rounded-md font-medium bg-[rgba(246,70,93,0.12)] text-[#f6465d] hover:bg-[rgba(246,70,93,0.2)] disabled:opacity-50 transition-colors"
                      >
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M18 15l-6-6-6 6"/></svg>
                        {commandMutation.isPending && commandMutation.variables?.workerId === w.worker_id ? "..." : "Unlist"}
                      </button>
                    ) : (
                      <button
                        disabled={commandMutation.isPending && commandMutation.variables?.workerId === w.worker_id}
                        onClick={() => setConfirmAction({ workerId: w.worker_id, targetState: "AVAILABLE" })}
                        className="inline-flex items-center gap-1 px-2.5 py-1 text-xs rounded-md font-medium bg-[rgba(14,203,129,0.12)] text-[#0ecb81] hover:bg-[rgba(14,203,129,0.2)] disabled:opacity-50 transition-colors"
                      >
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M6 9l6 6 6-6"/></svg>
                        {commandMutation.isPending && commandMutation.variables?.workerId === w.worker_id ? "..." : "List"}
                      </button>
                    )}
                    <span className={`inline-flex items-center gap-1.5 text-xs ${w.online ? "text-[#0ecb81]" : "text-[#f6465d]"}`}>
                      <span className="relative flex h-2 w-2">
                        {w.online && <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-[#0ecb81] opacity-75" />}
                        <span className={`relative inline-flex h-2 w-2 rounded-full ${w.online ? "bg-[#0ecb81]" : "bg-[#f6465d]"}`} />
                      </span>
                      {w.online ? "Online" : "Offline"}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <ConfirmDialog
        open={!!confirmAction}
        onOpenChange={(open) => { if (!open) setConfirmAction(null); }}
        title={confirmAction?.targetState === "AVAILABLE" ? "List Worker" : "Unlist Worker"}
        description={confirmAction?.targetState === "AVAILABLE"
          ? `List worker ${confirmAction.workerId} on the marketplace? It will be available for rent.`
          : `Unlist worker ${confirmAction?.workerId} from the marketplace? It will resume self-mining.`}
        confirmLabel={confirmAction?.targetState === "AVAILABLE" ? "List" : "Unlist"}
        variant={confirmAction?.targetState === "AVAILABLE" ? "primary" : "danger"}
        onConfirm={() => {
          if (confirmAction) commandMutation.mutate(confirmAction);
          setConfirmAction(null);
        }}
      />
    </div>
  );
}
