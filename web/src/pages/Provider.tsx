import { useState, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import type { UTCTimestamp } from "lightweight-charts";
import { tw, colors } from "../design/tokens";
import MetricCard from "../design/MetricCard";
import StatusBadge from "../design/StatusBadge";
import GpuBadge from "../design/GpuBadge";
import { ChartCard, Skeleton } from "../design";
import LWChart from "../design/LWChart";
import EmptyState from "../design/EmptyState";
import ConfirmDialog from "../design/ConfirmDialog";
import ViewToggle from "../design/ViewToggle";
import type { ViewMode } from "../design/ViewToggle";
import { usePersistedState } from "../hooks/usePersistedState";
import { useWallet } from "../context/WalletContext";
import { apiFetch } from "../api";
import { formatHashrate, formatUptime } from "../utils/format";
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

function PriceEditForm({
  workerId,
  currentPrice,
  onClose,
}: {
  workerId: string;
  currentPrice: number;
  onClose: () => void;
}) {
  const queryClient = useQueryClient();
  const [price, setPrice] = useState(currentPrice.toFixed(4));

  const priceMutation = useMutation({
    mutationFn: (price_per_min: number) =>
      apiFetch(`/api/provider/workers/${workerId}/pricing`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ price_per_min }),
      }),
    onSuccess: () => {
      toast.success("Price updated");
      queryClient.invalidateQueries({ queryKey: ["provider", "workers"] });
      onClose();
    },
    onError: () => toast.error("Failed to update price"),
  });

  const handleSave = () => {
    const val = parseFloat(price);
    if (isNaN(val) || val < 0) {
      toast.error("Invalid price");
      return;
    }
    priceMutation.mutate(val);
  };

  return (
    <div
      className="flex items-center gap-2"
      onClick={(e) => e.stopPropagation()}
    >
      <input
        type="number"
        step="0.0001"
        min="0"
        value={price}
        onChange={(e) => setPrice(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") handleSave();
          if (e.key === "Escape") onClose();
        }}
        autoFocus
        className={`${tw.input} w-28 py-1 px-2 text-xs font-mono`}
      />
      <button
        onClick={handleSave}
        disabled={priceMutation.isPending}
        className="px-2 py-1 text-xs rounded-md font-medium transition-colors disabled:opacity-50"
        style={{ background: colors.accent.muted, color: colors.accent.DEFAULT }}
      >
        {priceMutation.isPending ? "..." : "Save"}
      </button>
      <button
        onClick={onClose}
        disabled={priceMutation.isPending}
        className={`px-2 py-1 text-xs rounded-md font-medium ${tw.textTertiary} hover:${tw.textSecondary} transition-colors`}
      >
        Cancel
      </button>
    </div>
  );
}

export default function Provider() {
  const { address, connect } = useWallet();
  const queryClient = useQueryClient();

  const [viewWindow, setViewWindow] = useState(7 * 86400);
  const [filter, setFilter] = useState<FilterTab>("ALL");
  const [confirmAction, setConfirmAction] = useState<{
    workerId: string;
    targetState: "AVAILABLE" | "SELF_MINING";
  } | null>(null);
  const [viewMode, setViewMode] = usePersistedState<ViewMode>("provider-workers-view", "grid");
  const [editingPriceId, setEditingPriceId] = useState<string | null>(null);

  const { data: workersData, isLoading: workersLoading } = useQuery({
    queryKey: ["provider", "workers", address],
    queryFn: () => apiFetch<{ items: MyWorker[] }>("/api/provider/workers"),
    refetchInterval: 10_000,
    enabled: !!address,
  });
  const workers = workersData?.items ?? [];

  const { data: dashboard, isLoading: dashboardLoading } = useQuery({
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

  const chartData = useMemo(
    () => history.map((h) => ({ time: h.timestamp as UTCTimestamp, value: h.hashrate })),
    [history],
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

  const metricsLoading = dashboardLoading || !dashboard;
  const totalEarned = dashboard?.total_earned ?? 0;
  const totalBlocks = achievements?.total_blocks ?? dashboard?.total_blocks_mined ?? 0;
  const avgHashrate = dashboard?.avg_hashrate ?? 0;
  const peakHashrate = achievements?.peak_hashrate ?? 0;
  const miningDays = achievements?.mining_days ?? 0;

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
          aria-label="Share mining stats"
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
        <MetricCard label="Total Blocks" value={totalBlocks.toLocaleString()} variant="accent" loading={metricsLoading} />
        <MetricCard label="Peak Hashrate" value={formatHashrate(peakHashrate)} variant="info" loading={metricsLoading} />
        <MetricCard label="Total Earned" value={`${totalEarned.toFixed(2)} XNM`} variant="success" loading={metricsLoading} />
        <MetricCard label="Mining Days" value={miningDays.toFixed(0)} variant="warning" loading={metricsLoading} />
        <div
          className={`${tw.card} ${tw.cardHover} p-5 border-t-2 border-t-[#5e6673]`}
        >
          <span className={`text-xs ${tw.textTertiary} uppercase tracking-wider`}>Workers</span>
          {workersLoading ? (
            <Skeleton className="h-7 w-16 mt-1" />
          ) : (
            <div className={`text-2xl font-bold ${tw.textPrimary} mt-1 tabular-nums`}>
              {workers.filter(w => w.online).length}/{workers.length}
            </div>
          )}
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
            <span className={`text-xs ${tw.textTertiary}`}>0 H/s</span>
            <span className={`text-xs ${tw.textTertiary}`}>Peak: {formatHashrate(peakHashrate)}</span>
          </div>
        </div>
      </div>

      {/* My Workers */}
      <div className="min-h-[400px]">
        <div className="flex items-center justify-between mb-3">
          <h3 className={tw.sectionTitle}>My Workers ({workers.length})</h3>
          <ViewToggle value={viewMode} onChange={setViewMode} />
        </div>
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
        {workersLoading ? (
          viewMode === "grid" ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
              {Array.from({ length: 6 }).map((_, i) => (
                <Skeleton key={i} variant="card" className="h-48" />
              ))}
            </div>
          ) : (
            <div className={`${tw.card} overflow-x-auto`}>
              <table className="w-full text-sm">
                <thead>
                  <tr className={`${tw.surface2} border-b border-[#2a3441]`}>
                    <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Worker</th>
                    <th className={`${tw.tableHeader} px-4 py-3 text-left`}>GPU</th>
                    <th className={`${tw.tableHeader} px-4 py-3 text-right`}>Hashrate</th>
                    <th className={`${tw.tableHeader} px-4 py-3 text-right`}>Blocks</th>
                    <th className={`${tw.tableHeader} px-4 py-3 text-right`}>Uptime</th>
                    <th className={`${tw.tableHeader} px-4 py-3 text-center`}>Status</th>
                    <th className={`${tw.tableHeader} px-4 py-3 text-right`}>Price</th>
                    <th className={`${tw.tableHeader} px-4 py-3 text-right`}>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {Array.from({ length: 5 }).map((_, i) => (
                    <tr key={i} className="border-b border-[#1f2835]">
                      <td className="px-4 py-3"><Skeleton className="h-4 w-24" /></td>
                      <td className="px-4 py-3"><Skeleton className="h-4 w-28" /></td>
                      <td className="px-4 py-3"><Skeleton className="h-4 w-20 ml-auto" /></td>
                      <td className="px-4 py-3"><Skeleton className="h-4 w-10 ml-auto" /></td>
                      <td className="px-4 py-3"><Skeleton className="h-4 w-14 ml-auto" /></td>
                      <td className="px-4 py-3"><Skeleton className="h-5 w-16 rounded-full mx-auto" /></td>
                      <td className="px-4 py-3"><Skeleton className="h-4 w-16 ml-auto" /></td>
                      <td className="px-4 py-3"><Skeleton className="h-6 w-14 rounded-md ml-auto" /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        ) : filteredWorkers.length === 0 ? (
          <EmptyState
            title={filter === "ALL" ? "No workers found" : `No ${filterTabs.find((t) => t.key === filter)?.label.toLowerCase()} workers`}
            description={filter === "ALL" ? "Make sure your miners are registered with this wallet address." : "Try a different filter."}
          />
        ) : viewMode === "grid" ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
            {filteredWorkers.map((w) => (
              <div
                key={w.worker_id}
                className={`${tw.card} ${tw.cardHover} p-5 relative`}
                style={w.online ? {
                  boxShadow: `0 0 12px ${colors.success.muted}, inset 0 0 12px ${colors.success.muted}`,
                } : undefined}
              >
                {/* Header: worker id + status */}
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="relative flex h-2 w-2 shrink-0">
                      {w.online && <span className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-75" style={{ background: colors.success.DEFAULT }} />}
                      <span className={`relative inline-flex h-2 w-2 rounded-full`} style={{ background: w.online ? colors.success.DEFAULT : colors.danger.DEFAULT }} />
                    </span>
                    <span className={`font-mono text-sm font-medium ${tw.textPrimary} truncate`} title={w.worker_id}>{w.worker_id}</span>
                  </div>
                  <StatusBadge status={stateToStatus[w.state] || "idle"} label={stateToLabel[w.state]} size="sm" />
                </div>

                {/* GPU */}
                <div className="mb-3">
                  <GpuBadge name={w.gpu_name || "GPU"} memory={w.memory_gb || 0} />
                </div>

                {/* Price display */}
                <div
                  className="mb-3 py-2 px-3 rounded-md flex items-center justify-between"
                  style={{ background: colors.warning.muted }}
                >
                  {editingPriceId === w.worker_id ? (
                    <PriceEditForm
                      workerId={w.worker_id}
                      currentPrice={w.price_per_min}
                      onClose={() => setEditingPriceId(null)}
                    />
                  ) : (
                    <>
                      <span className="font-mono text-sm font-semibold tabular-nums" style={{ color: colors.warning.DEFAULT }}>
                        {w.price_per_min.toFixed(4)} <span className={`text-xs font-normal ${tw.textTertiary}`}>XNB/min</span>
                      </span>
                      {w.state !== "LEASED" && (
                        <button
                          onClick={() => setEditingPriceId(w.worker_id)}
                          className={`text-xs font-medium px-1.5 py-0.5 rounded transition-colors`}
                          style={{ color: colors.text.secondary }}
                          onMouseEnter={(e) => (e.currentTarget.style.color = colors.accent.DEFAULT)}
                          onMouseLeave={(e) => (e.currentTarget.style.color = colors.text.secondary)}
                        >
                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="inline -mt-px mr-0.5"><path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
                          Edit
                        </button>
                      )}
                    </>
                  )}
                </div>

                {/* Metrics */}
                <div className="grid grid-cols-3 gap-3 text-sm">
                  <div>
                    <p className={`text-xs ${tw.textTertiary} mb-0.5`}>Hashrate</p>
                    <p className={`font-mono font-semibold tabular-nums`} style={{ color: w.online ? colors.accent.DEFAULT : colors.text.primary }}>
                      {formatHashrate(w.hashrate)}
                    </p>
                  </div>
                  <div>
                    <p className={`text-xs ${tw.textTertiary} mb-0.5`}>Blocks</p>
                    <p className={`font-mono tabular-nums ${tw.textPrimary}`}>{w.self_blocks_found}</p>
                  </div>
                  <div>
                    <p className={`text-xs ${tw.textTertiary} mb-0.5`}>Uptime</p>
                    <p className={`tabular-nums ${tw.textPrimary}`}>{formatUptime(w.total_online_sec)}</p>
                  </div>
                </div>

                {/* Actions */}
                <div className="mt-3 pt-3 border-t border-[#1f2835] flex items-center justify-end gap-2">
                  {w.state === "LEASED" ? (
                    <span className={`text-xs font-medium ${tw.textTertiary}`}>Leased</span>
                  ) : w.state === "AVAILABLE" ? (
                    <button
                      disabled={commandMutation.isPending && commandMutation.variables?.workerId === w.worker_id}
                      onClick={() => setConfirmAction({ workerId: w.worker_id, targetState: "SELF_MINING" })}
                      className="inline-flex items-center gap-1 px-2.5 py-1 text-xs rounded-md font-medium disabled:opacity-50 transition-colors"
                      style={{ background: colors.danger.muted, color: colors.danger.DEFAULT }}
                    >
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M18 15l-6-6-6 6"/></svg>
                      {commandMutation.isPending && commandMutation.variables?.workerId === w.worker_id ? "..." : "Unlist"}
                    </button>
                  ) : (
                    <button
                      disabled={commandMutation.isPending && commandMutation.variables?.workerId === w.worker_id}
                      onClick={() => setConfirmAction({ workerId: w.worker_id, targetState: "AVAILABLE" })}
                      className="inline-flex items-center gap-1 px-2.5 py-1 text-xs rounded-md font-medium disabled:opacity-50 transition-colors"
                      style={{ background: colors.success.muted, color: colors.success.DEFAULT }}
                    >
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M6 9l6 6 6-6"/></svg>
                      {commandMutation.isPending && commandMutation.variables?.workerId === w.worker_id ? "..." : "List"}
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className={`${tw.card} overflow-x-auto`}>
            <table className="w-full text-sm">
              <thead>
                <tr className={`${tw.surface2} border-b border-[#2a3441]`}>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Worker</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left`}>GPU</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-right`}>Hashrate</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-right`}>Blocks</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-right`}>Uptime</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-center`}>Status</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-right`}>Price</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-right`}>Action</th>
                </tr>
              </thead>
              <tbody>
                {filteredWorkers.map((w) => (
                  <tr
                    key={w.worker_id}
                    className={tw.tableRow}
                    style={w.online ? { background: `${colors.success.DEFAULT}06` } : undefined}
                  >
                    <td className={`${tw.tableCell} font-mono text-xs`}>
                      <div className="flex items-center gap-2">
                        <span className={w.online ? tw.dotOnline : tw.dotOffline} />
                        <span className="truncate max-w-[120px] block" title={w.worker_id}>{w.worker_id}</span>
                      </div>
                    </td>
                    <td className={tw.tableCell}>
                      <GpuBadge name={w.gpu_name || "GPU"} memory={w.memory_gb || 0} />
                    </td>
                    <td className={`${tw.tableCell} font-mono tabular-nums text-right`}>
                      <span style={{ color: w.online ? colors.accent.DEFAULT : colors.text.primary }}>
                        {formatHashrate(w.hashrate)}
                      </span>
                    </td>
                    <td className={`${tw.tableCell} tabular-nums text-right`}>{w.self_blocks_found}</td>
                    <td className={`${tw.tableCell} tabular-nums text-right`}>{formatUptime(w.total_online_sec)}</td>
                    <td className={`${tw.tableCell} text-center`}>
                      <StatusBadge status={stateToStatus[w.state] || "idle"} label={stateToLabel[w.state]} size="sm" />
                    </td>
                    <td className={`${tw.tableCell} text-right`}>
                      {editingPriceId === w.worker_id ? (
                        <PriceEditForm
                          workerId={w.worker_id}
                          currentPrice={w.price_per_min}
                          onClose={() => setEditingPriceId(null)}
                        />
                      ) : (
                        <div className="flex items-center justify-end gap-1.5">
                          <span className="font-mono tabular-nums" style={{ color: colors.warning.DEFAULT }}>
                            {w.price_per_min.toFixed(4)}
                          </span>
                          <span className={`text-xs ${tw.textTertiary}`}>/min</span>
                          {w.state !== "LEASED" && (
                            <button
                              onClick={() => setEditingPriceId(w.worker_id)}
                              className="ml-1 p-0.5 rounded transition-colors"
                              style={{ color: colors.text.tertiary }}
                              onMouseEnter={(e) => (e.currentTarget.style.color = colors.accent.DEFAULT)}
                              onMouseLeave={(e) => (e.currentTarget.style.color = colors.text.tertiary)}
                              title="Edit price"
                            >
                              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
                            </button>
                          )}
                        </div>
                      )}
                    </td>
                    <td className={`${tw.tableCell} text-right`}>
                      {w.state === "LEASED" ? (
                        <span className={`text-xs ${tw.textTertiary}`}>Leased</span>
                      ) : w.state === "AVAILABLE" ? (
                        <button
                          disabled={commandMutation.isPending && commandMutation.variables?.workerId === w.worker_id}
                          onClick={() => setConfirmAction({ workerId: w.worker_id, targetState: "SELF_MINING" })}
                          className="px-2 py-1 text-xs rounded-md font-medium disabled:opacity-50 transition-colors"
                          style={{ background: colors.danger.muted, color: colors.danger.DEFAULT }}
                        >
                          Unlist
                        </button>
                      ) : (
                        <button
                          disabled={commandMutation.isPending && commandMutation.variables?.workerId === w.worker_id}
                          onClick={() => setConfirmAction({ workerId: w.worker_id, targetState: "AVAILABLE" })}
                          className="px-2 py-1 text-xs rounded-md font-medium disabled:opacity-50 transition-colors"
                          style={{ background: colors.success.muted, color: colors.success.DEFAULT }}
                        >
                          List
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
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
