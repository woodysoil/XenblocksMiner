import { useEffect, useState } from "react";
import { tw, colors } from "../design/tokens";
import MetricCard from "../design/MetricCard";
import StatusBadge from "../design/StatusBadge";
import GpuBadge from "../design/GpuBadge";
import HashText from "../design/HashText";
import DataTable, { type Column } from "../design/DataTable";

interface MyWorker {
  worker_id: string;
  gpu_name: string;
  memory_gb: number;
  hashrate: number;
  state: string;
  price_per_min: number;
  blocks_mined: number;
  current_lease: string | null;
  online_time?: string;
}

interface Lease {
  lease_id: string;
  consumer: string;
  duration_min: number;
  blocks: number;
  payout: number;
  status: string;
}

interface Dashboard {
  total_earned: number;
  week_earned: number;
  week_delta: number;
  active_leases: number;
  completion_rate: number;
}

// TODO: replace with API fetch
const fallbackWorkers: MyWorker[] = [
  { worker_id: "gpu-node-001", gpu_name: "RTX 4090", memory_gb: 24, hashrate: 1250, state: "SELF_MINING", price_per_min: 0.005, blocks_mined: 42, current_lease: null, online_time: "14d 6h" },
  { worker_id: "gpu-node-002", gpu_name: "RTX 3090", memory_gb: 24, hashrate: 890, state: "AVAILABLE", price_per_min: 0.003, blocks_mined: 28, current_lease: null, online_time: "7d 12h" },
  { worker_id: "gpu-node-003", gpu_name: "A100", memory_gb: 80, hashrate: 2100, state: "LEASED", price_per_min: 0.008, blocks_mined: 67, current_lease: "lease-abc123", online_time: "30d 1h" },
];

const fallbackLeases: Lease[] = [
  { lease_id: "lease-001", consumer: "0xabc1234567890def", duration_min: 60, blocks: 5, payout: 0.3, status: "completed" },
  { lease_id: "lease-002", consumer: "0x1234567890abcdef", duration_min: 120, blocks: 12, payout: 0.72, status: "completed" },
  { lease_id: "lease-abc123", consumer: "0xfedcba0987654321", duration_min: 45, blocks: 3, payout: 0.0, status: "active" },
];

function formatHashrate(h: number): string {
  if (h >= 1e6) return (h / 1e6).toFixed(2) + " MH/s";
  if (h >= 1e3) return (h / 1e3).toFixed(2) + " KH/s";
  return h.toFixed(1) + " H/s";
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
    key: "consumer",
    label: "Consumer",
    render: (v) => <HashText text={String(v)} chars={8} mono />,
  },
  {
    key: "duration_min",
    label: "Duration",
    render: (v) => <>{v} min</>,
  },
  { key: "blocks", label: "Blocks" },
  {
    key: "payout",
    label: "Payout",
    render: (v) => (
      <span className="font-mono" style={{ color: colors.success.DEFAULT }}>
        {Number(v).toFixed(4)} XNM
      </span>
    ),
  },
  {
    key: "status",
    label: "Status",
    render: (v) => {
      const s = String(v);
      const cls =
        s === "completed"
          ? tw.badgeSuccess
          : s === "active"
            ? tw.badgeAccent
            : `${tw.badgeDefault} bg-[#1f2835] text-[#5e6673]`;
      return <span className={cls}>{s}</span>;
    },
  },
];

export default function Provider() {
  const [workers, setWorkers] = useState<MyWorker[]>(fallbackWorkers);
  const [leases, setLeases] = useState<Lease[]>(fallbackLeases);
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);

  useEffect(() => {
    fetch("/api/provider/workers")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((d) => setWorkers(d.workers || d || fallbackWorkers))
      .catch(() => {});
    fetch("/api/provider/dashboard")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((d) => setDashboard(d))
      .catch(() => {});
    fetch("/api/provider/earnings")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((d) => setLeases(d.leases || d || fallbackLeases))
      .catch(() => {});
  }, []);

  const totalEarned = dashboard?.total_earned ?? leases.reduce((s, l) => s + l.payout, 0);
  const weekEarned = dashboard?.week_earned ?? 0.72;
  const weekDelta = dashboard?.week_delta ?? 12.5;
  const activeLeases = dashboard?.active_leases ?? leases.filter((l) => l.status === "active").length;
  const completed = leases.filter((l) => l.status === "completed").length;
  const rate = dashboard?.completion_rate ?? (leases.length ? Math.round((completed / leases.length) * 100) : 0);

  return (
    <div className="space-y-6">
      <h2 className={`text-xl font-semibold ${tw.textPrimary}`}>Provider Dashboard</h2>

      {/* Earnings cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard label="Total Earned" value={`${totalEarned.toFixed(2)} XNM`} variant="success" />
        <MetricCard
          label="This Week"
          value={`${weekEarned.toFixed(2)} XNM`}
          delta={`${weekDelta >= 0 ? "+" : ""}${weekDelta.toFixed(1)}%`}
          variant="accent"
        />
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
          <span className={`text-xs ${tw.textTertiary} uppercase tracking-wider`}>Completion Rate</span>
          <div className={`text-2xl font-bold ${tw.textPrimary} mt-1`}>{rate}%</div>
          <div className="h-1 rounded bg-[#1f2835] mt-2">
            <div
              className="h-1 rounded transition-all"
              style={{ width: `${rate}%`, backgroundColor: colors.success.DEFAULT }}
            />
          </div>
        </div>
      </div>

      {/* My Workers */}
      <div>
        <h3 className={`${tw.sectionTitle} mb-3`}>My Workers</h3>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {workers.map((w) => (
            <div key={w.worker_id} className={`${tw.card} p-5`}>
              <div className="flex items-center justify-between mb-3">
                <GpuBadge name={w.gpu_name} memory={w.memory_gb} />
                <StatusBadge status={stateToStatus[w.state] || "idle"} size="sm" />
              </div>
              <div className="grid grid-cols-3 gap-3 text-sm">
                <div>
                  <p className={`text-xs ${tw.textTertiary}`}>Hashrate</p>
                  <p className={`font-mono font-semibold ${tw.textPrimary}`}>{formatHashrate(w.hashrate)}</p>
                </div>
                <div>
                  <p className={`text-xs ${tw.textTertiary}`}>Blocks</p>
                  <p className={`font-mono ${tw.textPrimary}`}>{w.blocks_mined}</p>
                </div>
                <div>
                  <p className={`text-xs ${tw.textTertiary}`}>Online</p>
                  <p className={tw.textPrimary}>{w.online_time || "—"}</p>
                </div>
              </div>
              <div className="mt-3 pt-3 border-t border-[#1f2835] flex items-center justify-between">
                <span className="text-xs font-mono" style={{ color: colors.warning.DEFAULT }}>
                  {w.price_per_min.toFixed(4)} <span className={tw.textTertiary}>/min</span>
                </span>
                <button
                  className={`${tw.textTertiary} hover:${tw.textPrimary} transition-colors`}
                  aria-label="Edit price"
                >
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M17 3a2.85 2.85 0 114 4L7.5 20.5 2 22l1.5-5.5z" />
                  </svg>
                </button>
              </div>
              {w.current_lease && (
                <div className={`mt-2 px-3 py-2 rounded text-xs ${tw.surface2} flex items-center gap-2`}>
                  <span className={tw.textTertiary}>Leased to</span>
                  <HashText text={w.current_lease} chars={12} mono />
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Lease History */}
      <div>
        <h3 className={`${tw.sectionTitle} mb-3`}>Recent Leases</h3>
        <DataTable columns={leaseColumns} data={leases as unknown as Record<string, unknown>[]} />
      </div>
    </div>
  );
}
