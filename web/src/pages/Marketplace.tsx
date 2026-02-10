import { useEffect, useState } from "react";
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
  hashrate: number;
  price_per_min: number;
  blocks_mined: number;
  state: string;
  reputation: number;
}

function Stars({ score }: { score: number }) {
  const n = Math.min(Math.max(Math.round(score), 0), 5);
  return (
    <span className="inline-flex items-center gap-0.5 text-xs">
      {Array.from({ length: 5 }, (_, i) => (
        <span key={i} style={{ color: i < n ? colors.warning.DEFAULT : colors.border.default }}>
          {i < n ? "★" : "☆"}
        </span>
      ))}
    </span>
  );
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

  const rentMutation = useMutation({
    mutationFn: (body: { worker_id: string; duration_sec: number }) =>
      apiFetch("/api/rent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      toast.success("Rental started");
      queryClient.invalidateQueries({ queryKey: ["marketplace"] });
      setRentTarget(null);
    },
    onError: (err: Error) => {
      toast.error(err.message || "Rental failed");
    },
  });

  const openRentDialog = (p: ProviderListing) => {
    if (!address) { connect(); return; }
    setRentDuration(DURATION_OPTIONS[0].sec);
    setRentTarget(p);
  };

  const handleFilterChange = <T,>(setter: React.Dispatch<React.SetStateAction<T>>, value: T) => {
    setter(value);
    setPage(1);
  };

  const isAvailable = (state: string) => state === "IDLE" || state === "AVAILABLE";

  return (
    <div className="space-y-6">
      {/* Active Rentals collapsible */}
      <div className={`${tw.card} overflow-hidden`}>
        <button
          onClick={() => setRentalsOpen(!rentalsOpen)}
          className={`w-full flex items-center justify-between px-5 py-3 ${tw.textPrimary} text-sm font-medium hover:bg-[#1a2029]/80 active:bg-[#1f2835] transition-colors rounded-[10px]`}
          aria-expanded={rentalsOpen}
          aria-label="Toggle active rentals"
        >
          <span>Active Rentals (0)</span>
          <svg
            className={`w-4 h-4 transition-transform ${rentalsOpen ? "rotate-180" : ""}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </button>
        {rentalsOpen && (
          <div className="px-5 pb-4 border-t border-[#2a3441]">
            <EmptyState title="No active rentals" description="Rent hashpower from a provider below to get started" />
          </div>
        )}
      </div>

      {/* Header + Filters */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <h2 className={`text-xl font-semibold ${tw.textPrimary}`}>Hashpower Marketplace</h2>
        <div className="flex flex-wrap gap-3 items-center">
          <select value={gpuFilter} onChange={(e) => handleFilterChange(setGpuFilter, e.target.value)} className={`${tw.input} appearance-none pr-8 bg-[url('data:image/svg+xml;charset=utf-8,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2216%22%20height%3D%2216%22%20fill%3D%22%23848e9c%22%20viewBox%3D%220%200%2024%2024%22%3E%3Cpath%20d%3D%22M7%2010l5%205%205-5z%22%2F%3E%3C%2Fsvg%3E')] bg-[length:16px] bg-[right_8px_center] bg-no-repeat`}>
            {gpuTypes.map((g) => (
              <option key={g} value={g}>
                {g === "all" ? "All GPUs" : g}
              </option>
            ))}
          </select>
          <select value={sort} onChange={(e) => handleFilterChange(setSort, e.target.value)} className={`${tw.input} appearance-none pr-8 bg-[url('data:image/svg+xml;charset=utf-8,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2216%22%20height%3D%2216%22%20fill%3D%22%23848e9c%22%20viewBox%3D%220%200%2024%2024%22%3E%3Cpath%20d%3D%22M7%2010l5%205%205-5z%22%2F%3E%3C%2Fsvg%3E')] bg-[length:16px] bg-[right_8px_center] bg-no-repeat`}>
            <option value="price_asc">Price: Low → High</option>
            <option value="hashrate_desc">Hashrate: High → Low</option>
            <option value="reputation">Reputation</option>
          </select>
          <div className="relative">
            <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#5e6673] pointer-events-none" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
              <circle cx="11" cy="11" r="8" />
              <path d="m21 21-4.35-4.35" />
            </svg>
            <input
              type="text"
              placeholder="Search..."
              value={search}
              onChange={(e) => handleFilterChange(setSearch, e.target.value)}
              className={`${tw.input} pl-9`}
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
            <table className="w-full text-sm min-w-[700px]">
              <thead>
                <tr className={`${tw.surface2} border-b border-[#2a3441]`}>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Worker</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left`}>GPU</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Hashrate</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Price</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Blocks</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Rating</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Status</th>
                  <th className={`${tw.tableHeader} px-4 py-3 text-left`}></th>
                </tr>
              </thead>
              <tbody>
                {Array.from({ length: 5 }).map((_, i) => (
                  <tr key={i} className="border-b border-[#1f2835]">
                    <td className="px-4 py-3"><Skeleton className="h-4 w-24" /></td>
                    <td className="px-4 py-3"><Skeleton className="h-4 w-28" /></td>
                    <td className="px-4 py-3"><Skeleton className="h-4 w-20" /></td>
                    <td className="px-4 py-3"><Skeleton className="h-4 w-16" /></td>
                    <td className="px-4 py-3"><Skeleton className="h-4 w-10" /></td>
                    <td className="px-4 py-3"><Skeleton className="h-4 w-20" /></td>
                    <td className="px-4 py-3"><Skeleton className="h-4 w-16" /></td>
                    <td className="px-4 py-3"><Skeleton className="h-6 w-14 rounded-md" /></td>
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
              {providers.map((p) => (
                <div key={p.worker_id} className={`${tw.card} ${tw.cardHover} p-5 group hover:-translate-y-0.5 hover:shadow-lg transition-all duration-150`}>
                  <div className="flex items-center justify-between">
                    <HashText text={p.worker_id} chars={12} copyable />
                    <Stars score={p.reputation || 0} />
                  </div>

                  <div className="mt-3">
                    <GpuBadge name={p.gpu_count > 1 ? `${p.gpu_count}x ${p.gpu_name || "GPU"}` : p.gpu_name || "GPU"} />
                  </div>

                  <div className="grid grid-cols-3 gap-3 mt-4">
                    <div>
                      <p className={`text-sm font-semibold tabular-nums ${tw.textPrimary}`}>{formatHashrateCompact(p.hashrate)}</p>
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

                  <div className="mt-4 pt-4 border-t border-[#1f2835] flex items-center justify-between">
                    <span className="inline-flex items-center gap-1.5 text-xs">
                      <span className="relative flex h-2 w-2">
                        {isAvailable(p.state) && <span className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-75" style={{ backgroundColor: colors.success.DEFAULT }} />}
                        <span
                          className="relative inline-flex h-2 w-2 rounded-full"
                          style={{
                            backgroundColor: isAvailable(p.state) ? colors.success.DEFAULT : colors.text.tertiary,
                            boxShadow: isAvailable(p.state) ? `0 0 6px ${colors.success.DEFAULT}50` : undefined,
                          }}
                        />
                      </span>
                      <span style={{ color: isAvailable(p.state) ? colors.success.DEFAULT : colors.text.tertiary }}>
                        {isAvailable(p.state) ? "Available" : "Self-mining"}
                      </span>
                    </span>
                    <button
                      disabled={!isAvailable(p.state)}
                      onClick={() => openRentDialog(p)}
                      className={`${tw.btnPrimary} sm:opacity-0 sm:group-hover:opacity-100 transition-opacity disabled:opacity-40 disabled:cursor-not-allowed`}
                    >
                      Rent
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className={`${tw.card} overflow-x-auto`}>
              <table className="w-full text-sm min-w-[700px]">
                <thead>
                  <tr className={`${tw.surface2} border-b border-[#2a3441]`}>
                    <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Worker</th>
                    <th className={`${tw.tableHeader} px-4 py-3 text-left`}>GPU</th>
                    <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Hashrate</th>
                    <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Price</th>
                    <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Blocks</th>
                    <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Rating</th>
                    <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Status</th>
                    <th className={`${tw.tableHeader} px-4 py-3 text-left`}></th>
                  </tr>
                </thead>
                <tbody>
                  {providers.map((p) => (
                    <tr key={p.worker_id} className={tw.tableRow}>
                      <td className={`${tw.tableCell} font-mono text-xs`}>
                        <HashText text={p.worker_id} chars={12} copyable />
                      </td>
                      <td className={tw.tableCell}>
                        <GpuBadge name={p.gpu_count > 1 ? `${p.gpu_count}x ${p.gpu_name || "GPU"}` : p.gpu_name || "GPU"} />
                      </td>
                      <td className={`${tw.tableCell} font-mono tabular-nums`}>
                        {formatHashrateCompact(p.hashrate)} <span className={tw.textTertiary}>H/s</span>
                      </td>
                      <td className={`${tw.tableCell} font-mono tabular-nums`} style={{ color: colors.warning.DEFAULT }}>
                        {p.price_per_min.toFixed(4)} <span className={tw.textTertiary}>/min</span>
                      </td>
                      <td className={`${tw.tableCell} tabular-nums`}>{p.blocks_mined}</td>
                      <td className={tw.tableCell}><Stars score={p.reputation || 0} /></td>
                      <td className={tw.tableCell}>
                        <span className="inline-flex items-center gap-1.5 text-xs">
                          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: isAvailable(p.state) ? colors.success.DEFAULT : colors.text.tertiary }} />
                          <span style={{ color: isAvailable(p.state) ? colors.success.DEFAULT : colors.text.tertiary }}>
                            {isAvailable(p.state) ? "Available" : "Self-mining"}
                          </span>
                        </span>
                      </td>
                      <td className={tw.tableCell}>
                        <button
                          disabled={!isAvailable(p.state)}
                          onClick={() => openRentDialog(p)}
                          className={tw.btnPrimary + " text-xs px-3 py-1 disabled:opacity-40 disabled:cursor-not-allowed"}
                        >
                          Rent
                        </button>
                      </td>
                    </tr>
                  ))}
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
        description={
          rentTarget
            ? `Worker ${rentTarget.worker_id.slice(0, 12)}... — ${rentTarget.gpu_name || "GPU"} @ ${formatHashrateCompact(rentTarget.hashrate)} H/s`
            : ""
        }
        confirmLabel={rentMutation.isPending ? "Renting..." : "Confirm Rent"}
        confirmDisabled={rentMutation.isPending}
        onConfirm={() => {
          if (rentTarget) rentMutation.mutate({ worker_id: rentTarget.worker_id, duration_sec: rentDuration });
        }}
      >
        <div className="mt-4 space-y-3">
          <label className={`block text-xs font-medium ${tw.textSecondary}`}>Duration</label>
          <div className="flex gap-2">
            {DURATION_OPTIONS.map((opt) => (
              <button
                key={opt.sec}
                type="button"
                onClick={() => setRentDuration(opt.sec)}
                className={`flex-1 px-3 py-1.5 rounded-md text-xs font-medium border transition-colors ${
                  rentDuration === opt.sec
                    ? "bg-[#22d1ee]/15 border-[#22d1ee] text-[#22d1ee]"
                    : "border-[#2a3441] text-[#848e9c] hover:border-[#3d4f65]"
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
          {rentTarget && (
            <p className={`text-sm tabular-nums ${tw.textPrimary}`}>
              Estimated cost:{" "}
              <span style={{ color: colors.warning.DEFAULT }}>
                {(rentTarget.price_per_min * (rentDuration / 60)).toFixed(4)}
              </span>
            </p>
          )}
        </div>
      </ConfirmDialog>
    </div>
  );
}
