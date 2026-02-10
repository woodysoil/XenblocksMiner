import { tw } from "../design/tokens";
import EmptyState from "../design/EmptyState";
import { useWallet } from "../context/WalletContext";

export default function Account() {
  const { address, connect, disconnect } = useWallet();

  if (!address) {
    return (
      <EmptyState
        icon={
          <svg width="32" height="32" viewBox="0 0 20 20" fill="none" stroke="#22d1ee" strokeWidth="1.5">
            <circle cx="10" cy="7" r="4" />
            <path d="M3 18c0-3.3 3.1-6 7-6s7 2.7 7 6" />
          </svg>
        }
        title="Connect your wallet to view account settings"
        description="Link your Ethereum wallet to manage your account, API keys, and preferences."
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
      <h2 className={`text-xl font-semibold ${tw.textPrimary}`}>Account</h2>

      <div className={`${tw.card} p-5`}>
        <h3 className={`text-xs ${tw.textTertiary} uppercase tracking-wider mb-2`}>Wallet</h3>
        <p className={`font-mono text-sm ${tw.textPrimary} break-all`}>{address}</p>
      </div>

      <div className={`${tw.card} p-5`}>
        <h3 className={`text-xs ${tw.textTertiary} uppercase tracking-wider mb-2`}>API Keys</h3>
        <p className={`text-sm ${tw.textSecondary}`}>No API keys generated</p>
      </div>

      <button
        onClick={disconnect}
        className={`${tw.btnDanger} hover:shadow-[0_0_12px_rgba(246,70,93,0.2)] transition-all duration-200`}
      >
        Disconnect Wallet
      </button>
    </div>
  );
}
