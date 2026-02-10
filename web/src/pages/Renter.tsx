import { Link } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "../api";
import { toast } from "sonner";
import { tw } from "../design/tokens";
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

function stateBadge(state: string) {
  const s = state.toUpperCase();
  if (s === "ACTIVE") return <span className={tw.badgeSuccess}>Active</span>;
  if (s === "COMPLETED") return <span className={tw.badgeInfo}>Completed</span>;
  if (s === "CANCELLED") return <span className={tw.badgeDanger}>Cancelled</span>;
  return <span className={tw.badgeDefault + " bg-[#1f2835] text-[#848e9c]"}>{state}</span>;
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
  const hasLeases = leases.length > 0;

  return (
    <div className="space-y-6">
      <h2 className={`text-xl font-semibold ${tw.textPrimary}`}>Renter Dashboard</h2>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <MetricCard
          label="Active Leases"
          value={stats?.active_leases ?? 0}
          variant="accent"
          loading={statsLoading}
        />
        <MetricCard
          label="Total Spent"
          value={stats ? `${stats.total_spent.toFixed(4)} XNB` : "—"}
          variant="warning"
          loading={statsLoading}
        />
        <MetricCard
          label="Avg Cost"
          value={stats ? `${stats.avg_cost.toFixed(6)}/sec` : "—"}
          variant="info"
          loading={statsLoading}
        />
      </div>

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

      <div>
        <h3 className={`${tw.sectionTitle} mb-3`}>Lease History</h3>
        <div className={`${tw.card} overflow-hidden`}>
          <div className="overflow-x-auto">
            <table className="w-full text-sm min-w-[800px]">
              <thead>
                <tr className={`${tw.surface2} border-b border-[#2a3441]`}>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Lease</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Worker</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Duration</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Hashrate</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Blocks</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Cost</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Status</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Action</th>
                </tr>
              </thead>
              <tbody>
                {leasesLoading ? (
                  Array.from({ length: 3 }).map((_, i) => (
                    <tr key={i} className="border-b border-[#1f2835]">
                      <td className="px-4 py-3"><Skeleton className="h-4 w-20" /></td>
                      <td className="px-4 py-3"><Skeleton className="h-4 w-20" /></td>
                      <td className="px-4 py-3"><Skeleton className="h-4 w-16" /></td>
                      <td className="px-4 py-3"><Skeleton className="h-4 w-20" /></td>
                      <td className="px-4 py-3"><Skeleton className="h-4 w-10" /></td>
                      <td className="px-4 py-3"><Skeleton className="h-4 w-16" /></td>
                      <td className="px-4 py-3"><Skeleton className="h-5 w-16 rounded-full" /></td>
                      <td className="px-4 py-3"><Skeleton className="h-4 w-12" /></td>
                    </tr>
                  ))
                ) : leases.length === 0 ? (
                  <tr>
                    <td colSpan={8} className="py-12 text-center">
                      <p className={`text-sm ${tw.textSecondary}`}>No lease history yet</p>
                    </td>
                  </tr>
                ) : (
                  leases.map((l) => {
                    const cost = l.elapsed_sec * l.price_per_sec;
                    const active = l.state.toUpperCase() === "ACTIVE";
                    return (
                      <tr key={l.lease_id} className={tw.tableRow}>
                        <td className={tw.tableCell}>
                          <HashText text={l.lease_id} chars={8} copyable />
                        </td>
                        <td className={tw.tableCell}>
                          <HashText text={l.worker_id} chars={8} copyable />
                        </td>
                        <td className={`${tw.tableCell} tabular-nums`}>
                          {formatUptime(l.elapsed_sec)}
                        </td>
                        <td className={`${tw.tableCell} font-mono text-xs tabular-nums text-[#22d1ee]`}>
                          {l.avg_hashrate > 0 ? formatHashrate(l.avg_hashrate) : "—"}
                        </td>
                        <td className={`${tw.tableCell} tabular-nums`}>{l.blocks_found}</td>
                        <td className={`${tw.tableCell} tabular-nums`}>{cost.toFixed(4)}</td>
                        <td className={tw.tableCell}>{stateBadge(l.state)}</td>
                        <td className={tw.tableCell}>
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
