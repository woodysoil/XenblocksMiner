import { useEffect, useState, useCallback } from "react";
import { tw, colors } from "../design/tokens";

interface OverviewStats {
  total_workers: number;
  online_workers: number;
  total_hashrate: number;
  total_blocks: number;
  blocks_24h: number;
  active_leases: number;
  total_settled: number;
  platform_revenue: number;
}

interface ActivityItem {
  type: string;
  timestamp: number | string;
  details: Record<string, unknown>;
}

interface NetworkInfo {
  difficulty: number;
  total_workers: number;
  total_blocks: number;
  chain_blocks: number;
}

function formatHashrate(h: number): string {
  if (h >= 1e9) return (h / 1e9).toFixed(2) + " GH/s";
  if (h >= 1e6) return (h / 1e6).toFixed(2) + " MH/s";
  if (h >= 1e3) return (h / 1e3).toFixed(2) + " KH/s";
  return h.toFixed(1) + " H/s";
}

function timeAgo(ts: string | number): string {
  const ms = typeof ts === "number" ? (ts > 1e12 ? ts : ts * 1000) : new Date(ts).getTime();
  const sec = Math.max(0, Math.floor((Date.now() - ms) / 1000));
  if (sec < 60) return `${sec}s ago`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
  if (sec < 86400) return `${Math.floor(sec / 3600)}h ago`;
  return `${Math.floor(sec / 86400)}d ago`;
}

const eventDotColor: Record<string, string> = {
  block_found: colors.accent.DEFAULT,
  lease_started: colors.info.DEFAULT,
  lease_completed: colors.success.DEFAULT,
  worker_registered: colors.warning.DEFAULT,
};

const eventLabel = (a: ActivityItem): string => {
  const wid = a.details?.worker_id ? String(a.details.worker_id).slice(0, 12) : "";
  switch (a.type) {
    case "block_found": return `Block mined by ${wid || "unknown"}`;
    case "lease_started": return `Lease started${wid ? ` — ${wid}` : ""}`;
    case "lease_completed": return `Lease completed${wid ? ` — ${wid}` : ""}`;
    default: return `${a.type.replace(/_/g, " ")}${wid ? ` — ${wid}` : ""}`;
  }
};

export default function Overview() {
  const [stats, setStats] = useState<OverviewStats | null>(null);
  const [activity, setActivity] = useState<ActivityItem[]>([]);
  const [network, setNetwork] = useState<NetworkInfo | null>(null);

  const fetchAll = useCallback(() => {
    fetch("/api/overview/stats").then((r) => r.json()).then(setStats).catch(() => {});
    fetch("/api/overview/activity")
      .then((r) => r.json())
      .then((d) => setActivity(Array.isArray(d) ? d : d.items || []))
      .catch(() => {});
    fetch("/api/overview/network").then((r) => r.json()).then(setNetwork).catch(() => {});
  }, []);

  useEffect(() => {
    fetchAll();
    const id = setInterval(fetchAll, 30000);
    return () => clearInterval(id);
  }, [fetchAll]);

  const cards: { label: string; value: string | number; borderColor: string; icon: JSX.Element }[] = [
    {
      label: "Total Hashrate",
      value: stats ? formatHashrate(stats.total_hashrate) : "\u2014",
      borderColor: colors.accent.DEFAULT,
      icon: (
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="#22d1ee" strokeWidth="1.5">
          <path d="M10 2v4M10 14v4M4 8l2.5 4M13.5 8l2.5 4M6.5 12h7" />
          <circle cx="10" cy="10" r="8" strokeDasharray="2 2" opacity="0.3" />
        </svg>
      ),
    },
    {
      label: "Active Miners",
      value: stats?.online_workers ?? "\u2014",
      borderColor: colors.success.DEFAULT,
      icon: (
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
          <circle cx="10" cy="10" r="4" fill="#0ecb81" opacity="0.2" />
          <circle cx="10" cy="10" r="2" fill="#0ecb81" />
        </svg>
      ),
    },
    {
      label: "Blocks (24h)",
      value: stats?.blocks_24h ?? "\u2014",
      borderColor: colors.info.DEFAULT,
      icon: (
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="#3b82f6" strokeWidth="1.5">
          <rect x="3" y="3" width="14" height="14" rx="2" />
          <path d="M3 8h14M8 3v14" />
        </svg>
      ),
    },
    {
      label: "Revenue",
      value: stats ? `${stats.platform_revenue.toFixed(2)} XNM` : "\u2014",
      borderColor: colors.warning.DEFAULT,
      icon: (
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="#f0b90b" strokeWidth="1.5">
          <circle cx="10" cy="10" r="7" />
          <path d="M10 6v8M8 8h4M8 12h4" />
        </svg>
      ),
    },
  ];

  const networkItems = network
    ? [
        { label: "Difficulty", value: network.difficulty.toLocaleString() },
        { label: "Total Workers", value: network.total_workers.toLocaleString() },
        { label: "Total Blocks", value: network.total_blocks.toLocaleString() },
        { label: "Chain Blocks", value: network.chain_blocks.toLocaleString() },
      ]
    : [
        { label: "Difficulty", value: "\u2014" },
        { label: "Total Workers", value: "\u2014" },
        { label: "Total Blocks", value: "\u2014" },
        { label: "Chain Blocks", value: "\u2014" },
      ];

  return (
    <div className="space-y-6">
      {/* Stats row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {cards.map((c) => (
          <div
            key={c.label}
            className={`${tw.card} ${tw.cardHover} p-5`}
            style={{ borderTopWidth: 2, borderTopColor: c.borderColor }}
          >
            <div className="flex items-center justify-between">
              <span className={`text-xs ${tw.textTertiary} uppercase tracking-wider`}>{c.label}</span>
              {c.icon}
            </div>
            <div className={`text-2xl font-bold ${tw.textPrimary} mt-2`}>{c.value}</div>
          </div>
        ))}
      </div>

      {/* Activity + Network */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Activity Feed */}
        <div className={`lg:col-span-3 ${tw.card}`}>
          <div className="flex items-center justify-between px-5 pt-4 pb-3">
            <h3 className={`text-sm font-semibold ${tw.textPrimary}`}>Recent Activity</h3>
            <span className={`text-xs ${tw.textTertiary}`}>Last 24h</span>
          </div>
          <div className="px-5 pb-4 max-h-[400px] overflow-y-auto">
            {activity.length === 0 ? (
              <div className="flex items-center justify-center py-12">
                <span className={`text-sm ${tw.textSecondary}`}>No activity yet</span>
              </div>
            ) : (
              activity.map((a, i) => (
                <div
                  key={i}
                  className="flex items-center gap-3 py-2.5 border-b border-[#1f2835] text-sm"
                >
                  <span
                    className="w-2 h-2 rounded-full shrink-0"
                    style={{ backgroundColor: eventDotColor[a.type] || colors.text.tertiary }}
                  />
                  <span className={`flex-1 truncate ${tw.textPrimary}`}>{eventLabel(a)}</span>
                  <span className={`text-xs ${tw.textTertiary} shrink-0`}>{timeAgo(a.timestamp)}</span>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Network Status */}
        <div className={`lg:col-span-2 ${tw.card} p-5`}>
          <h3 className={`text-sm font-semibold ${tw.textPrimary} mb-4`}>Network Status</h3>
          {networkItems.map((item, i) => (
            <div
              key={item.label}
              className={`flex justify-between items-center py-3 ${
                i < networkItems.length - 1 ? "border-b border-[#1f2835]" : ""
              }`}
            >
              <span className="text-xs text-[#5e6673]">{item.label}</span>
              <span className="text-sm text-[#eaecef] font-medium">{item.value}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
