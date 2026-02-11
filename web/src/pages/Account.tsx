import { useState } from "react";
import { toast } from "sonner";
import { tw, colors } from "../design/tokens";
import { MetricCard, Skeleton } from "../design";
import EmptyState from "../design/EmptyState";
import { useWallet } from "../context/WalletContext";
import { apiFetch } from "../api";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";

interface AuthMe {
  account_id: string;
  role: string;
  eth_address: string;
  balance: number;
}

interface DepositPayload {
  amount: number;
}

interface WithdrawPayload {
  amount: number;
  eth_address?: string;
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <h3 className={`text-xs ${tw.textTertiary} uppercase tracking-wider mb-3`}>
      {children}
    </h3>
  );
}

function CopyButton({ text }: { text: string }) {
  return (
    <button
      onClick={() => {
        navigator.clipboard.writeText(text);
        toast.success("Copied");
      }}
      className={`shrink-0 p-2 rounded-md ${tw.textSecondary} hover:text-[${colors.accent.DEFAULT}] hover:bg-[${colors.bg.surface3}] transition-colors`}
      title="Copy"
      aria-label="Copy to clipboard"
    >
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="9" y="9" width="13" height="13" rx="2" />
        <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
      </svg>
    </button>
  );
}

function InlineForm({
  label,
  open,
  onToggle,
  onSubmit,
  isPending,
  children,
}: {
  label: string;
  open: boolean;
  onToggle: () => void;
  onSubmit: (e: React.FormEvent) => void;
  isPending: boolean;
  children: React.ReactNode;
}) {
  return (
    <div>
      <button onClick={onToggle} className={tw.btnSecondary}>
        {label}
      </button>
      {open && (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            onSubmit(e);
          }}
          className={`mt-3 p-4 rounded-lg bg-[${colors.bg.surface2}] border border-[${colors.border.default}] space-y-3`}
        >
          {children}
          <div className="flex gap-2">
            <button type="submit" disabled={isPending} className={tw.btnPrimary}>
              {isPending ? "Processing..." : "Confirm"}
            </button>
            <button type="button" onClick={onToggle} className={tw.btnSecondary}>
              Cancel
            </button>
          </div>
        </form>
      )}
    </div>
  );
}

export default function Account() {
  const { address, connect, disconnect } = useWallet();
  const queryClient = useQueryClient();

  const [depositOpen, setDepositOpen] = useState(false);
  const [withdrawOpen, setWithdrawOpen] = useState(false);
  const [depositAmt, setDepositAmt] = useState("");
  const [withdrawAmt, setWithdrawAmt] = useState("");
  const [withdrawAddr, setWithdrawAddr] = useState("");

  const { data: me, isLoading } = useQuery<AuthMe>({
    queryKey: ["auth-me"],
    queryFn: () => apiFetch<AuthMe>("/api/auth/me"),
    enabled: !!address,
  });

  const depositMutation = useMutation({
    mutationFn: (payload: DepositPayload) =>
      apiFetch(`/api/accounts/${me!.account_id}/deposit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
    onSuccess: () => {
      toast.success("Deposit successful");
      queryClient.invalidateQueries({ queryKey: ["auth-me"] });
      setDepositAmt("");
      setDepositOpen(false);
    },
    onError: (err: Error) => toast.error(err.message),
  });

  const withdrawMutation = useMutation({
    mutationFn: (payload: WithdrawPayload) =>
      apiFetch(`/api/accounts/${me!.account_id}/withdraw`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
    onSuccess: () => {
      toast.success("Withdrawal successful");
      queryClient.invalidateQueries({ queryKey: ["auth-me"] });
      setWithdrawAmt("");
      setWithdrawAddr("");
      setWithdrawOpen(false);
    },
    onError: (err: Error) => toast.error(err.message),
  });

  if (!address) {
    return (
      <EmptyState
        icon={
          <svg width="32" height="32" viewBox="0 0 20 20" fill="none" stroke={colors.accent.DEFAULT} strokeWidth="1.5">
            <circle cx="10" cy="7" r="4" />
            <path d="M3 18c0-3.3 3.1-6 7-6s7 2.7 7 6" />
          </svg>
        }
        title="Connect your wallet to view account settings"
        description="Link your Ethereum wallet to manage your account, API keys, and preferences."
        action={
          <button
            onClick={connect}
            className={`px-4 py-2 rounded-md bg-[${colors.accent.muted}] border border-[${colors.accent.DEFAULT}]/30 text-sm font-medium text-[${colors.accent.DEFAULT}] hover:bg-[${colors.accent.DEFAULT}]/20 transition-colors`}
          >
            Connect Wallet
          </button>
        }
      />
    );
  }

  const handleDeposit = () => {
    const amount = parseFloat(depositAmt);
    if (isNaN(amount) || amount <= 0) {
      toast.error("Enter a valid amount");
      return;
    }
    depositMutation.mutate({ amount });
  };

  const handleWithdraw = () => {
    const amount = parseFloat(withdrawAmt);
    if (isNaN(amount) || amount <= 0) {
      toast.error("Enter a valid amount");
      return;
    }
    const payload: WithdrawPayload = { amount };
    if (withdrawAddr.trim()) payload.eth_address = withdrawAddr.trim();
    withdrawMutation.mutate(payload);
  };

  const roleBadge = me?.role === "provider" ? tw.badgeSuccess : tw.badgeAccent;

  return (
    <div className="space-y-6">
      <h2 className={`text-xl font-semibold ${tw.textPrimary}`}>Account</h2>

      {/* Balance */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        <MetricCard
          label="Balance"
          value={isLoading ? "" : `${me?.balance ?? 0} XNB`}
          variant="accent"
          loading={isLoading}
          icon={
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2v20M17 5H9.5a3.5 3.5 0 000 7h5a3.5 3.5 0 010 7H6" />
            </svg>
          }
        />
        <MetricCard
          label="Role"
          value={isLoading ? "" : (me?.role ?? "-")}
          variant="info"
          loading={isLoading}
        />
      </div>

      {/* Deposit / Withdraw */}
      <div className={`${tw.card} p-5`}>
        <SectionLabel>Funds</SectionLabel>
        <div className="flex flex-wrap gap-3">
          <InlineForm
            label="Deposit"
            open={depositOpen}
            onToggle={() => {
              setDepositOpen((v) => !v);
              setWithdrawOpen(false);
            }}
            onSubmit={handleDeposit}
            isPending={depositMutation.isPending}
          >
            <input
              type="number"
              min="0"
              step="any"
              placeholder="Amount (XNB)"
              value={depositAmt}
              onChange={(e) => setDepositAmt(e.target.value)}
              className={`${tw.input} w-full`}
              required
            />
          </InlineForm>

          <InlineForm
            label="Withdraw"
            open={withdrawOpen}
            onToggle={() => {
              setWithdrawOpen((v) => !v);
              setDepositOpen(false);
            }}
            onSubmit={handleWithdraw}
            isPending={withdrawMutation.isPending}
          >
            <input
              type="number"
              min="0"
              step="any"
              placeholder="Amount (XNB)"
              value={withdrawAmt}
              onChange={(e) => setWithdrawAmt(e.target.value)}
              className={`${tw.input} w-full`}
              required
            />
            <input
              type="text"
              placeholder="Destination address (optional)"
              value={withdrawAddr}
              onChange={(e) => setWithdrawAddr(e.target.value)}
              className={`${tw.input} w-full`}
            />
          </InlineForm>
        </div>
      </div>

      {/* Transaction History */}
      <div className={`${tw.card} p-5`}>
        <SectionLabel>Transaction History</SectionLabel>
        <p className={`text-sm ${tw.textSecondary}`}>Transaction history coming soon</p>
      </div>

      {/* Account Info */}
      <div className={`${tw.card} p-5 space-y-4`}>
        <SectionLabel>Account Info</SectionLabel>
        {isLoading ? (
          <div className="space-y-3">
            <Skeleton className="h-4 w-48" />
            <Skeleton className="h-4 w-64" />
            <Skeleton className="h-4 w-56" />
          </div>
        ) : (
          <dl className="space-y-3 text-sm">
            <div className="flex items-center gap-2">
              <dt className={tw.textSecondary}>Role</dt>
              <dd><span className={roleBadge}>{me?.role ?? "-"}</span></dd>
            </div>
            <div className="flex items-baseline gap-2">
              <dt className={`${tw.textSecondary} shrink-0`}>Account ID</dt>
              <dd className={`font-mono ${tw.textPrimary} break-all`}>{me?.account_id ?? "-"}</dd>
            </div>
            <div className="flex items-center gap-2">
              <dt className={`${tw.textSecondary} shrink-0`}>Wallet</dt>
              <dd className={`font-mono ${tw.textPrimary} break-all flex-1`}>{address}</dd>
              <CopyButton text={address} />
            </div>
          </dl>
        )}
      </div>

      {/* API Keys */}
      <div className={`${tw.card} p-5`}>
        <SectionLabel>API Keys</SectionLabel>
        <div
          className={`flex items-center gap-3 p-3 rounded-lg bg-[${colors.bg.surface2}] border border-dashed border-[${colors.border.default}]`}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={colors.text.tertiary} strokeWidth="1.5">
            <path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 11-7.78 7.78 5.5 5.5 0 017.78-7.78zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4" />
          </svg>
          <p className={`text-sm ${tw.textSecondary}`}>No API keys generated</p>
        </div>
      </div>

      {/* Disconnect */}
      <button
        onClick={disconnect}
        className={`${tw.btnDanger} hover:shadow-[0_0_12px_${colors.danger.muted}] transition-all duration-200`}
      >
        Disconnect Wallet
      </button>
    </div>
  );
}
