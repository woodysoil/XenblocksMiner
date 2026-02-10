import { tw } from "../design/tokens";
import MetricCard from "../design/MetricCard";
import EmptyState from "../design/EmptyState";
import { useWallet } from "../context/WalletContext";

export default function Renter() {
  const { address, connect } = useWallet();

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

  return (
    <div className="space-y-6">
      <h2 className={`text-xl font-semibold ${tw.textPrimary}`}>Renter Dashboard</h2>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <MetricCard label="Active Leases" value="0" variant="accent" />
        <MetricCard label="Total Spent" value="0.00 XNM" variant="warning" />
        <MetricCard label="Avg Cost" value="—" variant="info" />
      </div>

      <div className={`${tw.card} p-6`}>
        <p className={`text-sm ${tw.textSecondary} text-center`}>
          No active leases — browse the Marketplace to rent hashpower
        </p>
      </div>

      <div>
        <h3 className={`${tw.sectionTitle} mb-3`}>Lease History</h3>
        <div className={tw.card}>
          <table className="w-full">
            <thead>
              <tr className="border-b border-[#1f2835]">
                <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Worker</th>
                <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Duration</th>
                <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Cost</th>
                <th className={`${tw.tableHeader} px-4 py-3 text-left`}>Status</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td colSpan={4} className="py-8">
                  <p className={`text-sm ${tw.textTertiary} text-center`}>No lease history</p>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
