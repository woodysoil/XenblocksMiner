import { useEffect, useState } from "react";
import { tw, colors } from "../design/tokens";
import GpuBadge from "../design/GpuBadge";
import HashText from "../design/HashText";
import EmptyState from "../design/EmptyState";

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

export default function Marketplace() {
  const [providers, setProviders] = useState<ProviderListing[]>([]);
  const [gpuFilter, setGpuFilter] = useState("all");
  const [sort, setSort] = useState("price_asc");
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [rentalsOpen, setRentalsOpen] = useState(false);

  useEffect(() => {
    fetch("/api/marketplace")
      .then((r) => r.json())
      .then((d) => setProviders(d.providers || d || []))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const gpuTypes = ["all", ...new Set(providers.map((p) => p.gpu_name).filter(Boolean))];

  const filtered = providers
    .filter((p) => gpuFilter === "all" || p.gpu_name === gpuFilter)
    .filter(
      (p) =>
        !search ||
        p.worker_id.toLowerCase().includes(search.toLowerCase()) ||
        (p.gpu_name || "").toLowerCase().includes(search.toLowerCase()),
    )
    .sort((a, b) => {
      if (sort === "price_asc") return a.price_per_min - b.price_per_min;
      if (sort === "hashrate_desc") return b.hashrate - a.hashrate;
      return (b.reputation || 0) - (a.reputation || 0);
    });

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
          <select value={gpuFilter} onChange={(e) => setGpuFilter(e.target.value)} className={tw.input}>
            {gpuTypes.map((g) => (
              <option key={g} value={g}>
                {g === "all" ? "All GPUs" : g}
              </option>
            ))}
          </select>
          <select value={sort} onChange={(e) => setSort(e.target.value)} className={tw.input}>
            <option value="price_asc">Price: Low → High</option>
            <option value="hashrate_desc">Hashrate: High → Low</option>
            <option value="reputation">Reputation</option>
          </select>
          <input
            type="text"
            placeholder="Search…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className={tw.input}
          />
        </div>
      </div>

      {/* Provider cards */}
      {loading ? (
        <div className="text-center py-16">
          <div className={`animate-pulse ${tw.textTertiary}`}>Loading providers…</div>
        </div>
      ) : filtered.length === 0 ? (
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
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
          {filtered.map((p) => (
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
      )}
    </div>
  );
}
