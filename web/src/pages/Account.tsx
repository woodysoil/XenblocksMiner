import { useState } from "react";
import { toast } from "sonner";
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
        <div className="flex items-center gap-3">
          <p className={`font-mono text-sm ${tw.textPrimary} break-all flex-1`}>{address}</p>
          <button
            onClick={() => {
              navigator.clipboard.writeText(address);
              toast.success("Address copied");
            }}
            className="shrink-0 p-2 rounded-md text-[#848e9c] hover:text-[#22d1ee] hover:bg-[#1f2835] transition-colors"
            title="Copy address"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="9" y="9" width="13" height="13" rx="2" />
              <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
            </svg>
          </button>
        </div>
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
