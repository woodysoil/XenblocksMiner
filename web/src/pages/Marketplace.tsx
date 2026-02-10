import { useEffect, useState, useMemo, useRef, useCallback } from "react";
import { tw, colors } from "../design/tokens";
import GpuBadge from "../design/GpuBadge";
import HashText from "../design/HashText";
import EmptyState from "../design/EmptyState";
import Pagination from "../components/Pagination";

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

function formatHashrate(h: number): string {
  if (h >= 1e6) return (h / 1e6).toFixed(2) + " MH";
  if (h >= 1e3) return (h / 1e3).toFixed(2) + " KH";
  return h.toFixed(1);
}

function Stars({ score }: { score: number }) {
  const n = Math.min(Math.max(Math.round(score), 0), 5);
  return (
    <span className="text-xs tracking-wide">
      {Array.from({ length: 5 }, (_, i) => (
        <span key={i} style={{ color: i < n ? colors.warning.DEFAULT : colors.border.default }}>
          {i < n ? "★" : "☆"}
        </span>
      ))}
    </span>
  );
}

const PAGE_SIZE = 18;

export default function Marketplace() {
  const [providers, setProviders] = useState<ProviderListing[]>([]);
  const [gpuTypes, setGpuTypes] = useState<string[]>(["all"]);
  const [totalPages, setTotalPages] = useState(1);
  const [gpuFilter, setGpuFilter] = useState("all");
  const [sort, setSort] = useState("price_asc");
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [rentalsOpen, setRentalsOpen] = useState(false);
  const [page, setPage] = useState(1);
  const isFilterChange = useRef(false);

  const fetchProviders = useCallback((p: number, sortBy: string, gpu: string, q: string) => {
    setLoading(true);
    const params = new URLSearchParams({
      page: String(p),
      limit: String(PAGE_SIZE),
      sort_by: sortBy,
    });
    if (gpu !== "all") params.set("gpu_type", gpu);
    if (q) params.set("search", q);

    fetch(`/api/marketplace?${params}`)
      .then((r) => r.json())
      .then((d) => {
        setProviders(d.items || []);
        setTotalPages(d.total_pages || 1);
        if (d.gpu_types) setGpuTypes(["all", ...d.gpu_types]);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetchProviders(page, sort, gpuFilter, search);
  }, [page, sort, gpuFilter, search, fetchProviders]);

  // Reset page when filters change
  useEffect(() => {
    if (isFilterChange.current) {
      setPage(1);
      isFilterChange.current = false;
    }
  }, [gpuFilter, sort, search]);

  const handleFilterChange = <T,>(setter: React.Dispatch<React.SetStateAction<T>>, value: T) => {
    isFilterChange.current = true;
    setter(value);
  };

  const isAvailable = (state: string) => state === "IDLE" || state === "AVAILABLE";

  return (
    <div className="space-y-6">
      {/* Active Rentals collapsible */}
      <div className={`${tw.card} overflow-hidden`}>
        <button
          onClick={() => setRentalsOpen(!rentalsOpen)}
          className={`w-full flex items-center justify-between px-5 py-3 ${tw.textPrimary} text-sm font-medium hover:bg-[#1a2029] transition-colors`}
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
          <select value={gpuFilter} onChange={(e) => handleFilterChange(setGpuFilter, e.target.value)} className={tw.input}>
            {gpuTypes.map((g) => (
              <option key={g} value={g}>
                {g === "all" ? "All GPUs" : g}
              </option>
            ))}
          </select>
          <select value={sort} onChange={(e) => handleFilterChange(setSort, e.target.value)} className={tw.input}>
            <option value="price_asc">Price: Low → High</option>
            <option value="hashrate_desc">Hashrate: High → Low</option>
            <option value="reputation">Reputation</option>
          </select>
          <input
            type="text"
            placeholder="Search…"
            value={search}
            onChange={(e) => handleFilterChange(setSearch, e.target.value)}
            className={tw.input}
          />
        </div>
      </div>

      {/* Provider cards */}
      {loading ? (
        <div className="text-center py-16">
          <div className={`animate-pulse ${tw.textTertiary}`}>Loading providers…</div>
        </div>
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
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
            {providers.map((p) => (
              <div key={p.worker_id} className={`${tw.card} ${tw.cardHover} p-5 group`}>
                <div className="flex items-center justify-between">
                  <HashText text={p.worker_id} chars={12} copyable />
                  <Stars score={p.reputation || 0} />
                </div>

                <div className="mt-3">
                  <GpuBadge name={p.gpu_count > 1 ? `${p.gpu_count}x ${p.gpu_name || "GPU"}` : p.gpu_name || "GPU"} />
                </div>

                <div className="grid grid-cols-3 gap-3 mt-4">
                  <div>
                    <p className={`text-sm font-semibold ${tw.textPrimary}`}>{formatHashrate(p.hashrate)}</p>
                    <p className={`text-xs ${tw.textTertiary}`}>H/s</p>
                  </div>
                  <div>
                    <p className="text-sm font-semibold" style={{ color: colors.warning.DEFAULT }}>
                      {p.price_per_min.toFixed(4)}
                    </p>
                    <p className={`text-xs ${tw.textTertiary}`}>/min</p>
                  </div>
                  <div>
                    <p className={`text-sm ${tw.textPrimary}`}>{p.blocks_mined}</p>
                    <p className={`text-xs ${tw.textTertiary}`}>mined</p>
                  </div>
                </div>

                <div className="mt-4 pt-4 border-t border-[#1f2835] flex items-center justify-between">
                  <span className="inline-flex items-center gap-1.5 text-xs">
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{
                        backgroundColor: isAvailable(p.state) ? colors.success.DEFAULT : colors.text.tertiary,
                        boxShadow: isAvailable(p.state) ? `0 0 6px ${colors.success.DEFAULT}50` : undefined,
                      }}
                    />
                    <span style={{ color: isAvailable(p.state) ? colors.success.DEFAULT : colors.text.tertiary }}>
                      {isAvailable(p.state) ? "Available" : "Self-mining"}
                    </span>
                  </span>
                  <button className={`${tw.btnPrimary} opacity-0 group-hover:opacity-100 transition-opacity`}>
                    Rent
                  </button>
                </div>
              </div>
            ))}
          </div>
          <Pagination currentPage={page} totalPages={totalPages} onPageChange={setPage} />
        </>
      )}
    </div>
  );
}
