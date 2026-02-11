import { Link } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useState, useEffect } from "react";
import { apiFetch } from "../api";
import { toast } from "sonner";
import { tw, colors } from "../design/tokens";
import MetricCard from "../design/MetricCard";
import EmptyState from "../design/EmptyState";
import { Skeleton, HashText } from "../design";
import { useWallet } from "../context/WalletContext";
import { formatHashrate, formatUptime } from "../utils/format";

interface RenterStats {
  active_leases: number;
  completed_leases: number;
  total_leases: number;
  total_spent: number;
  avg_cost: number;
}

interface Lease {
  lease_id: string;
  worker_id: string;
  consumer_id: string;
  mining_address?: string;
  duration_sec: number;
  price_per_sec: number;
  state: string;
  created_at: string;
  ended_at: string | null;
  blocks_found: number;
  avg_hashrate: number;
  elapsed_sec: number;
}

interface LeaseList {
  items: Lease[];
  total: number;
}

function formatRemaining(sec: number): string {
  if (sec <= 0) return "expiring";
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  if (h > 0) return `${h}h ${m}m remaining`;
  if (m > 0) return `${m}m remaining`;
  return `${sec}s remaining`;
}

function stateBadge(state: string) {
  const s = state.toUpperCase();
  if (s === "ACTIVE") return <span className={tw.badgeSuccess}>Active</span>;
  if (s === "COMPLETED") return <span className={tw.badgeInfo}>Completed</span>;
  if (s === "CANCELLED") return <span className={tw.badgeDanger}>Cancelled</span>;
  return <span className={tw.badgeDefault + " bg-[#1f2835] text-[#848e9c]"}>{state}</span>;
}

function ProgressBar({ elapsed, total }: { elapsed: number; total: number }) {
  const pct = total > 0 ? Math.min(100, (elapsed / total) * 100) : 0;
  return (
    <div
      className="w-full h-1.5 rounded-full overflow-hidden"
      style={{ backgroundColor: colors.accent.muted }}
    >
      <div
        className="h-full rounded-full transition-all duration-500"
        style={{
          width: `${pct}%`,
          background: `linear-gradient(90deg, ${colors.accent.DEFAULT}, ${colors.info.DEFAULT})`,
        }}
      />
    </div>
  );
}

function ActiveLeaseCard({
  lease,
  onStop,
  stopping,
}: {
  lease: Lease;
  onStop: (id: string) => void;
  stopping: boolean;
}) {
  const [elapsed, setElapsed] = useState(lease.elapsed_sec);

  useEffect(() => {
    setElapsed(lease.elapsed_sec);
    const interval = setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => clearInterval(interval);
  }, [lease.elapsed_sec]);

  const remaining = Math.max(0, lease.duration_sec - elapsed);
  const pct = lease.duration_sec > 0 ? Math.min(100, (elapsed / lease.duration_sec) * 100) : 0;
  const cost = elapsed * lease.price_per_sec;

  return (
    <div
      className={`${tw.card} p-4 border-t-2`}
      style={{
        borderTopColor: colors.accent.DEFAULT,
        boxShadow: `0 -1px 12px ${colors.accent.glow}`,
      }}
    >
      <div className="flex items-start justify-between gap-4 mb-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 mb-1">
            <span className={tw.dotActive} />
            <span className={`text-sm font-medium ${tw.textPrimary}`}>
              Worker <HashText text={lease.worker_id} chars={8} copyable />
            </span>
          </div>
          {lease.mining_address && (
            <div className={`text-xs ${tw.textSecondary} ml-4`}>
              Mining to <HashText text={lease.mining_address} chars={10} copyable />
            </div>
          )}
        </div>
        <button
          className={tw.btnDanger + " !px-3 !py-1.5 text-xs whitespace-nowrap"}
          disabled={stopping}
          onClick={() => onStop(lease.lease_id)}
        >
          {stopping ? "Stopping..." : "Stop Lease"}
        </button>
      </div>

      <div className="grid grid-cols-3 gap-3 mb-3">
        <div>
          <span className={`text-[10px] uppercase tracking-wider ${tw.textTertiary}`}>Hashrate</span>
          <div className="text-sm font-mono tabular-nums" style={{ color: colors.accent.DEFAULT }}>
            {lease.avg_hashrate > 0 ? formatHashrate(lease.avg_hashrate) : "--"}
          </div>
        </div>
        <div>
          <span className={`text-[10px] uppercase tracking-wider ${tw.textTertiary}`}>Blocks</span>
          <div className={`text-sm font-mono tabular-nums ${tw.textPrimary}`}>{lease.blocks_found}</div>
        </div>
        <div>
          <span className={`text-[10px] uppercase tracking-wider ${tw.textTertiary}`}>Cost</span>
          <div className={`text-sm font-mono tabular-nums ${tw.textPrimary}`}>{cost.toFixed(4)} XNB</div>
        </div>
      </div>

      <div className="space-y-1.5">
        <div className="flex items-center justify-between text-xs">
          <span className={tw.textSecondary}>{pct.toFixed(0)}% elapsed</span>
          <span style={{ color: colors.accent.DEFAULT }}>{formatRemaining(remaining)}</span>
        </div>
        <ProgressBar elapsed={elapsed} total={lease.duration_sec} />
      </div>
    </div>
  );
}

export default function Renter() {
  const { address, connect } = useWallet();
  const queryClient = useQueryClient();

  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ["renter", "stats"],
    queryFn: () => apiFetch<RenterStats>("/api/renter/stats"),
    enabled: !!address,
  });

  const { data: leaseData, isLoading: leasesLoading } = useQuery({
    queryKey: ["renter", "leases"],
    queryFn: () => apiFetch<LeaseList>("/api/renter/leases"),
    enabled: !!address,
    refetchInterval: 15_000,
  });

  const stopMutation = useMutation({
    mutationFn: (lease_id: string) =>
      apiFetch("/api/stop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lease_id }),
      }),
    onSuccess: () => {
      toast.success("Lease stopped");
      queryClient.invalidateQueries({ queryKey: ["renter"] });
    },
    onError: (err: Error) => toast.error(err.message),
  });

  if (!address) {
    return (
      <EmptyState
        icon={
          <svg width="32" height="32" viewBox="0 0 20 20" fill="none" stroke="#22d1ee" strokeWidth="1.5">
            <rect x="2" y="4" width="16" height="12" rx="2" />
            <path d="M2 8h16" />
          </svg>
        }
        title="Connect your wallet to manage rentals"
        description="Link your Ethereum wallet to rent hashpower, track leases, and view spending history."
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

  const leases = leaseData?.items ?? [];
  const activeLeases = leases.filter((l) => l.state.toUpperCase() === "ACTIVE");
  const historyLeases = leases.filter((l) => l.state.toUpperCase() !== "ACTIVE");
  const hasLeases = leases.length > 0;
  const totalBlocks = leases.reduce((sum, l) => sum + l.blocks_found, 0);

  return (
    <div className="space-y-6">
      <h2 className={`text-xl font-semibold ${tw.textPrimary}`}>Renter Dashboard</h2>

      {/* Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="Active Leases"
          value={stats?.active_leases ?? 0}
          variant="accent"
          loading={statsLoading}
        />
        <MetricCard
          label="Total Spent"
          value={stats ? `${stats.total_spent.toFixed(4)} XNB` : "\u2014"}
          variant="warning"
          loading={statsLoading}
        />
        <MetricCard
          label="Avg Cost"
          value={stats ? `${stats.avg_cost.toFixed(6)}/sec` : "\u2014"}
          variant="info"
          loading={statsLoading}
        />
        <MetricCard
          label="Blocks Mined"
          value={leasesLoading ? 0 : totalBlocks}
          delta={stats ? `${stats.completed_leases} completed leases` : undefined}
          variant="success"
          loading={statsLoading || leasesLoading}
        />
      </div>

      {/* Empty state */}
      {!hasLeases && !leasesLoading && (
        <div className={`${tw.card} p-8 text-center`}>
          <h3 className={`text-base font-semibold ${tw.textPrimary} mb-2`}>Ready to rent hashpower?</h3>
          <p className={`text-sm ${tw.textSecondary} mb-4`}>
            Browse available miners on the Marketplace and start your first lease.
          </p>
          <Link to="/marketplace" className={tw.btnPrimary}>
            Browse Marketplace
          </Link>
        </div>
      )}

      {/* Active Leases Section */}
      {(activeLeases.length > 0 || leasesLoading) && (
        <div>
          <h3 className={`${tw.sectionTitle} mb-3 flex items-center gap-2`}>
            <span className={tw.dotActive} />
            Active Leases
          </h3>
          {leasesLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Array.from({ length: 2 }).map((_, i) => (
                <Skeleton key={i} variant="card" className="h-40" />
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {activeLeases.map((l) => (
                <ActiveLeaseCard
                  key={l.lease_id}
                  lease={l}
                  onStop={(id) => stopMutation.mutate(id)}
                  stopping={stopMutation.isPending}
                />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Lease History Table */}
      <div>
        <h3 className={`${tw.sectionTitle} mb-3`}>Lease History</h3>
        <div className={`${tw.card} overflow-hidden`}>
          <div className="overflow-x-auto">
            <table className="w-full text-sm min-w-[900px]" style={{ fontVariantNumeric: "tabular-nums" }}>
              <thead>
                <tr className={`${tw.surface2} border-b border-[#2a3441]`}>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left w-[14%]`}>Lease</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left w-[14%]`}>Worker</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left w-[14%]`}>Mining To</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left w-[12%]`}>Duration</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-right w-[12%]`}>Hashrate</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-right w-[8%]`}>Blocks</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-right w-[10%]`}>Cost</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-center w-[10%]`}>Status</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-right w-[6%]`}>Action</th>
                </tr>
              </thead>
              <tbody>
                {leasesLoading ? (
                  Array.from({ length: 3 }).map((_, i) => (
                    <tr key={i} className="border-b border-[#1f2835]">
                      <td className="px-4 py-3"><Skeleton className="h-4 w-20" /></td>
                      <td className="px-4 py-3"><Skeleton className="h-4 w-20" /></td>
                      <td className="px-4 py-3"><Skeleton className="h-4 w-24" /></td>
                      <td className="px-4 py-3"><Skeleton className="h-4 w-16" /></td>
                      <td className="px-4 py-3"><Skeleton className="h-4 w-20" /></td>
                      <td className="px-4 py-3"><Skeleton className="h-4 w-10" /></td>
                      <td className="px-4 py-3"><Skeleton className="h-4 w-16" /></td>
                      <td className="px-4 py-3"><Skeleton className="h-5 w-16 rounded-full" /></td>
                      <td className="px-4 py-3"><Skeleton className="h-4 w-12" /></td>
                    </tr>
                  ))
                ) : historyLeases.length === 0 ? (
                  <tr>
                    <td colSpan={9} className="py-12 text-center">
                      <p className={`text-sm ${tw.textSecondary}`}>No lease history yet</p>
                    </td>
                  </tr>
                ) : (
                  historyLeases.map((l) => {
                    const cost = l.elapsed_sec * l.price_per_sec;
                    const active = l.state.toUpperCase() === "ACTIVE";
                    const pct = l.duration_sec > 0 ? Math.min(100, (l.elapsed_sec / l.duration_sec) * 100) : 0;
                    return (
                      <tr key={l.lease_id} className={tw.tableRow}>
                        <td className={tw.tableCell}>
                          <HashText text={l.lease_id} chars={8} copyable />
                        </td>
                        <td className={tw.tableCell}>
                          <HashText text={l.worker_id} chars={8} copyable />
                        </td>
                        <td className={tw.tableCell}>
                          {l.mining_address ? (
                            <HashText text={l.mining_address} chars={10} copyable />
                          ) : (
                            <span className={tw.textTertiary}>--</span>
                          )}
                        </td>
                        <td className={tw.tableCell}>
                          <div className="space-y-1">
                            <span>{formatUptime(l.elapsed_sec)}</span>
                            {l.duration_sec > 0 && (
                              <ProgressBar elapsed={l.elapsed_sec} total={l.duration_sec} />
                            )}
                            <span className={`block text-[10px] ${tw.textTertiary}`}>
                              {pct.toFixed(0)}% of {formatUptime(l.duration_sec)}
                            </span>
                          </div>
                        </td>
                        <td className={`${tw.tableCell} text-right font-mono text-xs`} style={{ color: colors.accent.DEFAULT }}>
                          {l.avg_hashrate > 0 ? formatHashrate(l.avg_hashrate) : "\u2014"}
                        </td>
                        <td className={`${tw.tableCell} text-right`}>{l.blocks_found}</td>
                        <td className={`${tw.tableCell} text-right`}>{cost.toFixed(4)}</td>
                        <td className={`${tw.tableCell} text-center`}>{stateBadge(l.state)}</td>
                        <td className={`${tw.tableCell} text-right`}>
                          {active && (
                            <button
                              className={tw.btnDanger + " !px-3 !py-1 text-xs"}
                              disabled={stopMutation.isPending}
                              onClick={() => stopMutation.mutate(l.lease_id)}
                            >
                              Stop
                            </button>
                          )}
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
