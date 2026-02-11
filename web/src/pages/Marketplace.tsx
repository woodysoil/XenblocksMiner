import { useEffect, useState, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "../api";
import { toast } from "sonner";
import { useWallet } from "../context/WalletContext";
import { tw, colors } from "../design/tokens";
import { Skeleton } from "../design";
import GpuBadge from "../design/GpuBadge";
import HashText from "../design/HashText";
import EmptyState from "../design/EmptyState";
import ConfirmDialog from "../design/ConfirmDialog";
import Pagination from "../components/Pagination";
import ViewToggle from "../design/ViewToggle";
import type { ViewMode } from "../design/ViewToggle";
import { usePersistedState } from "../hooks/usePersistedState";
import { formatHashrateCompact } from "../utils/format";

interface ProviderListing {
  worker_id: string;
  gpu_name: string;
  gpu_count: number;
  gpu_memory?: number;
  hashrate: number;
  price_per_min: number;
  blocks_mined: number;
  state: string;
  reputation: number;
  uptime_pct?: number;
}

interface ActiveLease {
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
  items: ActiveLease[];
  total: number;
}

function Stars({ score }: { score: number }) {
  const n = Math.min(Math.max(Math.round(score), 0), 5);
  return (
    <span className="inline-flex items-center gap-0.5 text-xs">
      {Array.from({ length: 5 }, (_, i) => (
        <span key={i} style={{ color: i < n ? colors.warning.DEFAULT : colors.border.default }}>
          {i < n ? "\u2605" : "\u2606"}
        </span>
      ))}
    </span>
  );
}

function UptimeBar({ pct }: { pct: number }) {
  const clamped = Math.min(Math.max(pct, 0), 100);
  const barColor =
    clamped >= 95 ? colors.success.DEFAULT : clamped >= 80 ? colors.warning.DEFAULT : colors.danger.DEFAULT;
  return (
    <div className="flex items-center gap-1.5">
      <div
        className="h-1.5 rounded-full flex-1 max-w-[48px]"
        style={{ backgroundColor: colors.bg.surface3 }}
      >
        <div
          className="h-full rounded-full transition-all duration-300"
          style={{ width: `${clamped}%`, backgroundColor: barColor }}
        />
      </div>
      <span className={`text-[10px] tabular-nums ${tw.textTertiary}`}>{clamped.toFixed(0)}%</span>
    </div>
  );
}

function formatTimeRemaining(durationSec: number, elapsedSec: number): string {
  const rem = Math.max(0, durationSec - elapsedSec);
  if (rem <= 0) return "Expired";
  const h = Math.floor(rem / 3600);
  const m = Math.floor((rem % 3600) / 60);
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

const DURATION_OPTIONS = [
  { label: "1 hour", sec: 3600 },
  { label: "6 hours", sec: 21600 },
  { label: "24 hours", sec: 86400 },
] as const;

const PAGE_SIZE = 18;

export default function Marketplace() {
  const { address, connect } = useWallet();
  const queryClient = useQueryClient();

  const [viewMode, setViewMode] = usePersistedState<ViewMode>("marketplace-view", "grid");
  const [gpuFilter, setGpuFilter] = useState("all");
  const [sort, setSort] = useState("price_asc");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [rentalsOpen, setRentalsOpen] = useState(false);
  const [gpuTypes, setGpuTypes] = useState<string[]>(["all"]);

  const [rentTarget, setRentTarget] = useState<ProviderListing | null>(null);
  const [rentDuration, setRentDuration] = useState<number>(DURATION_OPTIONS[0].sec);
  const [miningAddress, setMiningAddress] = useState("");

  // Fetch marketplace listings
  const { data, isLoading: loading } = useQuery({
    queryKey: ["marketplace", page, sort, gpuFilter, search],
    queryFn: () => {
      const params = new URLSearchParams({
        page: String(page),
        limit: String(PAGE_SIZE),
        sort_by: sort,
      });
      if (gpuFilter !== "all") params.set("gpu_type", gpuFilter);
      if (search) params.set("search", search);
      return apiFetch<{ items: ProviderListing[]; total_pages: number; gpu_types?: string[] }>(
        `/api/marketplace?${params}`,
      );
    },
  });

  const providers = data?.items ?? [];
  const totalPages = data?.total_pages ?? 1;

  useEffect(() => {
    if (data?.gpu_types) setGpuTypes(["all", ...data.gpu_types]);
  }, [data?.gpu_types]);

  // Fetch active rentals for connected wallet
  const { data: leaseData, isLoading: leasesLoading } = useQuery({
    queryKey: ["renter", "leases", "active"],
    queryFn: () => apiFetch<LeaseList>("/api/renter/leases?state=active"),
    enabled: !!address,
    refetchInterval: 30_000,
  });

  const activeLeases = useMemo(
    () => (leaseData?.items ?? []).filter((l) => l.state.toUpperCase() === "ACTIVE"),
    [leaseData],
  );

  const rentMutation = useMutation({
    mutationFn: (body: { worker_id: string; duration_sec: number; consumer_address: string }) =>
      apiFetch("/api/rent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      toast.success("Rental started");
      queryClient.invalidateQueries({ queryKey: ["marketplace"] });
      queryClient.invalidateQueries({ queryKey: ["renter"] });
      setRentTarget(null);
    },
    onError: (err: Error) => {
      toast.error(err.message || "Rental failed");
    },
  });

  const openRentDialog = (p: ProviderListing) => {
    if (!address) { connect(); return; }
    setRentDuration(DURATION_OPTIONS[0].sec);
    setMiningAddress(address);
    setRentTarget(p);
  };

  const handleFilterChange = <T,>(setter: React.Dispatch<React.SetStateAction<T>>, value: T) => {
    setter(value);
    setPage(1);
  };

  const isAvailable = (state: string) => state === "IDLE" || state === "AVAILABLE";

  const estimatedCost = rentTarget ? rentTarget.price_per_min * (rentDuration / 60) : 0;
  const costPerHour = rentTarget ? rentTarget.price_per_min * 60 : 0;

  return (
    <div className="space-y-6">
      {/* Active Rentals collapsible */}
      <div className={`${tw.card} overflow-hidden`}>
        <button
          onClick={() => setRentalsOpen(!rentalsOpen)}
          className={`w-full flex items-center justify-between px-5 py-3 ${tw.textPrimary} text-sm font-medium hover:bg-[${colors.bg.surface2}]/80 active:bg-[${colors.bg.surface3}] transition-colors rounded-[10px]`}
          aria-expanded={rentalsOpen}
          aria-label="Toggle active rentals"
        >
          <span className="flex items-center gap-2">
            Active Rentals
            {address && (
              <span
                className="inline-flex items-center justify-center min-w-[20px] h-5 px-1.5 rounded-full text-xs font-semibold tabular-nums"
                style={{
                  backgroundColor: activeLeases.length > 0 ? colors.accent.muted : colors.bg.surface3,
                  color: activeLeases.length > 0 ? colors.accent.DEFAULT : colors.text.tertiary,
                }}
              >
                {leasesLoading ? "\u2014" : activeLeases.length}
              </span>
            )}
            {!address && (
              <span
                className="inline-flex items-center justify-center min-w-[20px] h-5 px-1.5 rounded-full text-xs font-semibold"
                style={{ backgroundColor: colors.bg.surface3, color: colors.text.tertiary }}
              >
                0
              </span>
            )}
          </span>
          <svg
            className={`w-4 h-4 transition-transform duration-200 ${rentalsOpen ? "rotate-180" : ""}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </button>
        <div
          className="overflow-hidden transition-all duration-200"
          style={{
            maxHeight: rentalsOpen ? "600px" : "0",
            opacity: rentalsOpen ? 1 : 0,
          }}
        >
          <div className="px-5 pb-4" style={{ borderTop: `1px solid ${colors.border.default}` }}>
            {!address ? (
              <EmptyState
                title="Wallet not connected"
                description="Connect your wallet to view active rentals"
                action={
                  <button onClick={connect} className={tw.btnPrimary}>
                    Connect Wallet
                  </button>
                }
              />
            ) : leasesLoading ? (
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3 pt-4">
                {Array.from({ length: 2 }).map((_, i) => (
                  <Skeleton key={i} variant="card" className="h-28" />
                ))}
              </div>
            ) : activeLeases.length === 0 ? (
              <EmptyState title="No active rentals" description="Rent hashpower from a provider below to get started" />
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3 pt-4">
                {activeLeases.map((l) => (
                  <div
                    key={l.lease_id}
                    className={`${tw.card} p-4`}
                    style={{ borderLeftWidth: 2, borderLeftColor: colors.accent.DEFAULT }}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <HashText text={l.worker_id} chars={10} copyable />
                      <span className={tw.badgeSuccess}>Active</span>
                    </div>
                    <div className="grid grid-cols-3 gap-2">
                      <div>
                        <p className={`text-xs ${tw.textTertiary}`}>Hashrate</p>
                        <p className={`text-sm font-semibold tabular-nums ${tw.textPrimary}`}>
                          {formatHashrateCompact(l.avg_hashrate)} <span className={`text-xs ${tw.textTertiary}`}>H/s</span>
                        </p>
                      </div>
                      <div>
                        <p className={`text-xs ${tw.textTertiary}`}>Remaining</p>
                        <p className={`text-sm font-semibold tabular-nums`} style={{ color: colors.accent.DEFAULT }}>
                          {formatTimeRemaining(l.duration_sec, l.elapsed_sec)}
                        </p>
                      </div>
                      <div>
                        <p className={`text-xs ${tw.textTertiary}`}>Blocks</p>
                        <p className={`text-sm font-semibold tabular-nums ${tw.textPrimary}`}>{l.blocks_found}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Header + Filters */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <h2 className={`text-xl font-semibold ${tw.textPrimary}`}>Hashpower Marketplace</h2>
        <div className="flex flex-wrap gap-3 items-center">
          <div className="relative">
            <select
              value={gpuFilter}
              onChange={(e) => handleFilterChange(setGpuFilter, e.target.value)}
              className={`${tw.input} appearance-none pr-8 cursor-pointer`}
              style={{
                backgroundImage: `url("data:image/svg+xml;charset=utf-8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='${encodeURIComponent(colors.text.tertiary)}' viewBox='0 0 24 24'%3E%3Cpath d='M7 10l5 5 5-5z'/%3E%3C/svg%3E")`,
                backgroundSize: "16px",
                backgroundPosition: "right 8px center",
                backgroundRepeat: "no-repeat",
              }}
            >
              {gpuTypes.map((g) => (
                <option key={g} value={g}>
                  {g === "all" ? "All GPUs" : g}
                </option>
              ))}
            </select>
          </div>
          <div className="relative">
            <select
              value={sort}
              onChange={(e) => handleFilterChange(setSort, e.target.value)}
              className={`${tw.input} appearance-none pr-8 cursor-pointer`}
              style={{
                backgroundImage: `url("data:image/svg+xml;charset=utf-8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='${encodeURIComponent(colors.text.tertiary)}' viewBox='0 0 24 24'%3E%3Cpath d='M7 10l5 5 5-5z'/%3E%3C/svg%3E")`,
                backgroundSize: "16px",
                backgroundPosition: "right 8px center",
                backgroundRepeat: "no-repeat",
              }}
            >
              <option value="price_asc">Price: Low &rarr; High</option>
              <option value="hashrate_desc">Hashrate: High &rarr; Low</option>
              <option value="reputation">Reputation</option>
            </select>
          </div>
          <div className="relative group">
            <svg
              className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 pointer-events-none transition-colors duration-150"
              style={{ color: colors.text.tertiary }}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth="2"
            >
              <circle cx="11" cy="11" r="8" />
              <path d="m21 21-4.35-4.35" />
            </svg>
            <input
              type="text"
              placeholder="Search providers..."
              value={search}
              onChange={(e) => handleFilterChange(setSearch, e.target.value)}
              className={`${tw.input} pl-9 w-48 focus:w-60 transition-all duration-200`}
            />
          </div>
          <ViewToggle value={viewMode} onChange={setViewMode} />
        </div>
      </div>

      {/* Provider cards */}
      {loading ? (
        viewMode === "grid" ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
            {Array.from({ length: 6 }).map((_, i) => (
              <Skeleton key={i} variant="card" className="h-48" />
            ))}
          </div>
        ) : (
          <div className={`${tw.card} overflow-x-auto`}>
            <table className="w-full text-sm min-w-[800px]">
              <thead>
                <tr className={`${tw.surface2} border-b border-[${colors.border.default}]`}>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left w-[15%]`}>Worker</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left w-[18%]`}>GPU</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left w-[12%]`}>Hashrate</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left w-[12%]`}>Price</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left w-[8%]`}>Blocks</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left w-[10%]`}>Rating</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left w-[12%]`}>Status</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-right w-[13%]`}></th>
                </tr>
              </thead>
              <tbody>
                {Array.from({ length: 5 }).map((_, i) => (
                  <tr key={i} style={{ borderBottom: `1px solid ${colors.bg.surface3}` }}>
                    <td className="px-4 py-3"><Skeleton className="h-4 w-24" /></td>
                    <td className="px-4 py-3"><Skeleton className="h-4 w-28" /></td>
                    <td className="px-4 py-3"><Skeleton className="h-4 w-20" /></td>
                    <td className="px-4 py-3"><Skeleton className="h-4 w-16" /></td>
                    <td className="px-4 py-3"><Skeleton className="h-4 w-10" /></td>
                    <td className="px-4 py-3"><Skeleton className="h-4 w-20" /></td>
                    <td className="px-4 py-3"><Skeleton className="h-4 w-16" /></td>
                    <td className="px-4 py-3 text-right"><Skeleton className="h-6 w-14 rounded-md ml-auto" /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )
      ) : providers.length === 0 ? (
        <EmptyState
          icon={
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z" />
              <polyline points="9,22 9,12 15,12 15,22" />
            </svg>
          }
          title="No providers available"
          description="Check back later or adjust your filters"
        />
      ) : (
        <>
          {viewMode === "grid" ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
              {providers.map((p) => {
                const available = isAvailable(p.state);
                return (
                  <div
                    key={p.worker_id}
                    className={`${tw.card} ${tw.cardHover} p-5 group hover:-translate-y-0.5 hover:shadow-lg transition-all duration-150`}
                    style={{
                      borderLeftWidth: available ? 2 : 1,
                      borderLeftColor: available ? colors.accent.DEFAULT : colors.border.default,
                      boxShadow: available ? `inset 2px 0 8px -4px ${colors.accent.glow}` : undefined,
                    }}
                  >
                    <div className="flex items-center justify-between">
                      <HashText text={p.worker_id} chars={12} copyable />
                      <Stars score={p.reputation || 0} />
                    </div>

                    <div className="mt-3 flex items-center gap-2">
                      <GpuBadge
                        name={p.gpu_count > 1 ? `${p.gpu_count}x ${p.gpu_name || "GPU"}` : p.gpu_name || "GPU"}
                        memory={p.gpu_memory}
                      />
                      {p.uptime_pct != null && <UptimeBar pct={p.uptime_pct} />}
                    </div>

                    <div className="grid grid-cols-3 gap-3 mt-4">
                      <div>
                        <p className={`text-sm font-semibold tabular-nums ${tw.textPrimary}`}>
                          {formatHashrateCompact(p.hashrate)}
                        </p>
                        <p className={`text-xs ${tw.textTertiary}`}>H/s</p>
                      </div>
                      <div>
                        <p className="text-sm font-semibold tabular-nums" style={{ color: colors.warning.DEFAULT }}>
                          {p.price_per_min.toFixed(4)}
                        </p>
                        <p className={`text-xs ${tw.textTertiary}`}>/min</p>
                      </div>
                      <div>
                        <p className={`text-sm tabular-nums ${tw.textPrimary}`}>{p.blocks_mined}</p>
                        <p className={`text-xs ${tw.textTertiary}`}>mined</p>
                      </div>
                    </div>

                    <div
                      className="mt-4 pt-4 flex items-center justify-between"
                      style={{ borderTop: `1px solid ${colors.bg.surface3}` }}
                    >
                      <span className="inline-flex items-center gap-1.5 text-xs">
                        <span className="relative flex h-2 w-2">
                          {available && (
                            <span
                              className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-75"
                              style={{ backgroundColor: colors.success.DEFAULT }}
                            />
                          )}
                          <span
                            className="relative inline-flex h-2 w-2 rounded-full"
                            style={{
                              backgroundColor: available ? colors.success.DEFAULT : colors.text.tertiary,
                              boxShadow: available ? `0 0 6px ${colors.success.DEFAULT}50` : undefined,
                            }}
                          />
                        </span>
                        <span style={{ color: available ? colors.success.DEFAULT : colors.text.tertiary }}>
                          {available ? "Available" : "Self-mining"}
                        </span>
                      </span>
                      <button
                        disabled={!available}
                        onClick={() => openRentDialog(p)}
                        className={`${tw.btnPrimary} sm:opacity-0 sm:group-hover:opacity-100 transition-opacity disabled:opacity-40 disabled:cursor-not-allowed`}
                      >
                        Rent
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className={`${tw.card} overflow-x-auto`}>
              <table className="w-full text-sm min-w-[800px]">
                <thead>
                  <tr className={`${tw.surface2}`} style={{ borderBottom: `1px solid ${colors.border.default}` }}>
                    <th className={`${tw.tableHeader} px-4 py-3 text-left w-[15%]`}>Worker</th>
                    <th className={`${tw.tableHeader} px-4 py-3 text-left w-[18%]`}>GPU</th>
                    <th className={`${tw.tableHeader} px-4 py-3 text-left w-[12%]`}>Hashrate</th>
                    <th className={`${tw.tableHeader} px-4 py-3 text-left w-[12%]`}>Price</th>
                    <th className={`${tw.tableHeader} px-4 py-3 text-left w-[8%]`}>Blocks</th>
                    <th className={`${tw.tableHeader} px-4 py-3 text-left w-[10%]`}>Rating</th>
                    <th className={`${tw.tableHeader} px-4 py-3 text-left w-[12%]`}>Status</th>
                    <th className={`${tw.tableHeader} px-4 py-3 text-right w-[13%]`}></th>
                  </tr>
                </thead>
                <tbody>
                  {providers.map((p) => {
                    const available = isAvailable(p.state);
                    return (
                      <tr
                        key={p.worker_id}
                        className={tw.tableRow}
                        style={{
                          borderLeftWidth: available ? 2 : 0,
                          borderLeftColor: available ? colors.accent.DEFAULT : "transparent",
                        }}
                      >
                        <td className={`${tw.tableCell} font-mono text-xs`}>
                          <HashText text={p.worker_id} chars={12} copyable />
                        </td>
                        <td className={tw.tableCell}>
                          <GpuBadge
                            name={p.gpu_count > 1 ? `${p.gpu_count}x ${p.gpu_name || "GPU"}` : p.gpu_name || "GPU"}
                            memory={p.gpu_memory}
                          />
                        </td>
                        <td className={`${tw.tableCell} font-mono tabular-nums`}>
                          {formatHashrateCompact(p.hashrate)}{" "}
                          <span className={tw.textTertiary}>H/s</span>
                        </td>
                        <td className={`${tw.tableCell} font-mono tabular-nums`} style={{ color: colors.warning.DEFAULT }}>
                          {p.price_per_min.toFixed(4)}{" "}
                          <span className={tw.textTertiary}>/min</span>
                        </td>
                        <td className={`${tw.tableCell} tabular-nums`}>{p.blocks_mined}</td>
                        <td className={tw.tableCell}><Stars score={p.reputation || 0} /></td>
                        <td className={tw.tableCell}>
                          <span className="inline-flex items-center gap-1.5 text-xs">
                            <span
                              className="w-2 h-2 rounded-full"
                              style={{ backgroundColor: available ? colors.success.DEFAULT : colors.text.tertiary }}
                            />
                            <span style={{ color: available ? colors.success.DEFAULT : colors.text.tertiary }}>
                              {available ? "Available" : "Self-mining"}
                            </span>
                          </span>
                        </td>
                        <td className={`${tw.tableCell} text-right`}>
                          <button
                            disabled={!available}
                            onClick={() => openRentDialog(p)}
                            className={`${tw.btnPrimary} text-xs !px-3 !py-1 disabled:opacity-40 disabled:cursor-not-allowed`}
                          >
                            Rent
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
          <Pagination currentPage={page} totalPages={totalPages} onPageChange={setPage} />
        </>
      )}

      {/* Rent confirmation dialog */}
      <ConfirmDialog
        open={!!rentTarget}
        onOpenChange={(open) => { if (!open) setRentTarget(null); }}
        title="Rent Hashpower"
        description=""
        confirmLabel={rentMutation.isPending ? "Renting..." : "Confirm Rent"}
        confirmDisabled={rentMutation.isPending || !miningAddress}
        onConfirm={() => {
          if (rentTarget) {
            rentMutation.mutate({
              worker_id: rentTarget.worker_id,
              duration_sec: rentDuration,
              consumer_address: miningAddress,
            });
          }
        }}
      >
        {rentTarget && (
          <div className="mt-4 space-y-4">
            {/* Provider summary */}
            <div
              className="rounded-lg p-3 flex items-center justify-between"
              style={{ backgroundColor: colors.bg.surface3, border: `1px solid ${colors.border.default}` }}
            >
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <GpuBadge
                    name={
                      rentTarget.gpu_count > 1
                        ? `${rentTarget.gpu_count}x ${rentTarget.gpu_name || "GPU"}`
                        : rentTarget.gpu_name || "GPU"
                    }
                    memory={rentTarget.gpu_memory}
                  />
                </div>
                <p className={`text-xs font-mono ${tw.textTertiary}`}>
                  {rentTarget.worker_id.slice(0, 16)}...
                </p>
              </div>
              <div className="text-right space-y-1">
                <p className={`text-sm font-semibold tabular-nums ${tw.textPrimary}`}>
                  {formatHashrateCompact(rentTarget.hashrate)} H/s
                </p>
                <Stars score={rentTarget.reputation || 0} />
              </div>
            </div>

            {/* Mining address */}
            <div>
              <label className={`block text-xs font-medium ${tw.textSecondary} mb-1.5`}>Mining Address</label>
              <input
                type="text"
                value={miningAddress}
                onChange={(e) => setMiningAddress(e.target.value)}
                placeholder="0x..."
                className={`${tw.input} w-full font-mono text-xs`}
              />
              <p className={`text-[10px] mt-1 ${tw.textTertiary}`}>
                Ethereum address where mined blocks will be credited
              </p>
            </div>

            {/* Duration selector */}
            <div>
              <label className={`block text-xs font-medium ${tw.textSecondary} mb-1.5`}>Duration</label>
              <div className="flex gap-2">
                {DURATION_OPTIONS.map((opt) => {
                  const selected = rentDuration === opt.sec;
                  const optCost = rentTarget.price_per_min * (opt.sec / 60);
                  return (
                    <button
                      key={opt.sec}
                      type="button"
                      onClick={() => setRentDuration(opt.sec)}
                      className="flex-1 rounded-md text-xs font-medium border transition-colors px-3 py-2"
                      style={{
                        backgroundColor: selected ? colors.accent.muted : "transparent",
                        borderColor: selected ? colors.accent.DEFAULT : colors.border.default,
                        color: selected ? colors.accent.DEFAULT : colors.text.secondary,
                      }}
                      onMouseEnter={(e) => {
                        if (!selected) e.currentTarget.style.borderColor = colors.border.hover;
                      }}
                      onMouseLeave={(e) => {
                        if (!selected) e.currentTarget.style.borderColor = colors.border.default;
                      }}
                    >
                      <div>{opt.label}</div>
                      <div
                        className="text-[10px] mt-0.5 tabular-nums"
                        style={{ color: selected ? colors.accent.hover : colors.text.tertiary }}
                      >
                        {optCost.toFixed(4)}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Cost breakdown */}
            <div
              className="rounded-lg p-3 space-y-1.5"
              style={{ backgroundColor: colors.bg.surface3, border: `1px solid ${colors.border.default}` }}
            >
              <div className="flex justify-between text-xs">
                <span className={tw.textSecondary}>Rate</span>
                <span className={tw.textPrimary} style={{ fontVariantNumeric: "tabular-nums" }}>
                  {rentTarget.price_per_min.toFixed(4)} / min
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className={tw.textSecondary}>Hourly</span>
                <span className={tw.textPrimary} style={{ fontVariantNumeric: "tabular-nums" }}>
                  {costPerHour.toFixed(4)} / hr
                </span>
              </div>
              <div
                className="flex justify-between text-sm font-semibold pt-1.5 mt-1.5"
                style={{ borderTop: `1px solid ${colors.border.default}` }}
              >
                <span className={tw.textPrimary}>Total</span>
                <span style={{ color: colors.warning.DEFAULT, fontVariantNumeric: "tabular-nums" }}>
                  {estimatedCost.toFixed(4)}
                </span>
              </div>
            </div>
          </div>
        )}
      </ConfirmDialog>
    </div>
  );
}
