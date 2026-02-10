import { useEffect, useState, useCallback } from "react";
import { tw, colors } from "../design/tokens";
import MetricCard from "../design/MetricCard";
import StatusBadge from "../design/StatusBadge";
import GpuBadge from "../design/GpuBadge";
import HashText from "../design/HashText";
import DataTable, { type Column } from "../design/DataTable";
import EmptyState from "../design/EmptyState";
import { useWallet } from "../context/WalletContext";
import { apiFetch } from "../api";

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

interface Lease {
  lease_id: string;
  consumer_id: string;
  duration_sec: number;
  blocks_found: number;
  provider_payment: number;
}

interface Dashboard {
  total_earned: number;
  active_leases: number;
  total_blocks_mined: number;
  avg_hashrate: number;
  worker_count: number;
}

function formatHashrate(h: number): string {
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

const leaseColumns: Column<Record<string, unknown>>[] = [
  {
    key: "lease_id",
    label: "Lease ID",
    render: (v) => <HashText text={String(v)} chars={12} mono />,
  },
  {
    key: "consumer_id",
    label: "Consumer",
    render: (v) => <HashText text={String(v)} chars={8} mono />,
  },
  {
    key: "duration_sec",
    label: "Duration",
    render: (v) => <>{Math.round(Number(v) / 60)} min</>,
  },
  { key: "blocks_found", label: "Blocks" },
  {
    key: "provider_payment",
    label: "Payout",
    render: (v) => (
      <span className="font-mono" style={{ color: colors.success.DEFAULT }}>
        {Number(v).toFixed(4)} XNM
      </span>
    ),
  },
];

export default function Provider() {
  const { address, connect } = useWallet();
  const [workers, setWorkers] = useState<MyWorker[]>([]);
  const [leases, setLeases] = useState<Lease[]>([]);
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);

  const fetchData = useCallback(() => {
    if (!address) return;
    apiFetch<{ items: MyWorker[] }>("/api/provider/workers")
      .then((d) => setWorkers(d.items || []))
      .catch(() => {});
    apiFetch<Dashboard>("/api/provider/dashboard")
      .then((d) => setDashboard(d))
      .catch(() => {});
    apiFetch<{ earnings: Lease[] }>("/api/provider/earnings")
      .then((d) => setLeases(d.earnings || []))
      .catch(() => {});
  }, [address]);

  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, 10000);
    return () => clearInterval(id);
  }, [fetchData]);

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
        description="Link your Ethereum wallet to manage workers, track earnings, and monitor leases."
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
  const activeLeases = dashboard?.active_leases ?? 0;
  const totalBlocks = dashboard?.total_blocks_mined ?? 0;
  const avgHashrate = dashboard?.avg_hashrate ?? 0;

  return (
    <div className="space-y-6">
      <h2 className={`text-xl font-semibold ${tw.textPrimary}`}>Provider Dashboard</h2>

      {/* Metric cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard label="Total Earned" value={`${totalEarned.toFixed(2)} XNM`} variant="success" />
        <MetricCard label="Workers" value={String(workers.length)} variant="accent" />
        <div
          className={`${tw.card} ${tw.cardHover} p-5 border-t-2`}
          style={{ borderTopColor: colors.info.DEFAULT }}
        >
          <span className={`text-xs ${tw.textTertiary} uppercase tracking-wider`}>Active Leases</span>
          <div className={`text-2xl font-bold ${tw.textPrimary} mt-1`}>{activeLeases}</div>
        </div>
        <div
          className={`${tw.card} ${tw.cardHover} p-5 border-t-2`}
          style={{ borderTopColor: colors.warning.DEFAULT }}
        >
          <span className={`text-xs ${tw.textTertiary} uppercase tracking-wider`}>Avg Hashrate</span>
          <div className={`text-2xl font-bold ${tw.textPrimary} mt-1`}>{formatHashrate(avgHashrate)}</div>
        </div>
      </div>

      {/* My Workers */}
      <div>
        <h3 className={`${tw.sectionTitle} mb-3`}>My Workers</h3>
        {workers.length === 0 ? (
          <EmptyState title="No workers found for this wallet" description="Make sure your miners are registered with this address." />
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
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

      {/* Lease History */}
      {leases.length > 0 && (
        <div>
          <h3 className={`${tw.sectionTitle} mb-3`}>Recent Leases</h3>
          <DataTable columns={leaseColumns} data={leases as unknown as Record<string, unknown>[]} />
        </div>
      )}
    </div>
  );
}
