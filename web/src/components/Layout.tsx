import { useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { useDashboard } from "../context/DashboardContext";
import { useWallet } from "../context/WalletContext";

const publicNav = [
  {
    to: "/",
    label: "Overview",
    icon: (
      <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
        <rect x="2" y="2" width="7" height="7" rx="1.5" />
        <rect x="11" y="2" width="7" height="7" rx="1.5" />
        <rect x="2" y="11" width="7" height="7" rx="1.5" />
        <rect x="11" y="11" width="7" height="7" rx="1.5" />
      </svg>
    ),
  },
  {
    to: "/monitoring",
    label: "Monitoring",
    icon: (
      <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
        <polyline points="1,10 4,10 7,4 10,16 13,7 16,10 19,10" />
      </svg>
    ),
  },
  {
    to: "/marketplace",
    label: "Marketplace",
    icon: (
      <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M3 7l1.5-4h11L17 7" />
        <rect x="3" y="7" width="14" height="10" rx="1.5" />
        <path d="M7 7v2a3 3 0 006 0V7" />
      </svg>
    ),
  },
];

const walletNav = [
  {
    to: "/provider",
    label: "Provider",
    icon: (
      <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M10 2l7 4v4l-7 4-7-4V6l7-4z" />
        <path d="M10 10v8" />
        <path d="M3 6l7 4 7-4" />
      </svg>
    ),
  },
  {
    to: "/renter",
    label: "Renter",
    icon: (
      <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
        <rect x="2" y="4" width="16" height="12" rx="2" />
        <path d="M2 8h16" />
      </svg>
    ),
  },
  {
    to: "/account",
    label: "Account",
    icon: (
      <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="10" cy="7" r="4" />
        <path d="M3 18c0-3.3 3.1-6 7-6s7 2.7 7 6" />
      </svg>
    ),
  },
];

const pageTitles: Record<string, string> = {
  "/": "Overview",
  "/monitoring": "Monitoring",
  "/marketplace": "Marketplace",
  "/provider": "Provider",
  "/renter": "Renter",
  "/account": "Account",
};

function SideNavLink({ to, label, icon, end, onClick }: { to: string; label: string; icon: React.ReactNode; end?: boolean; onClick?: () => void }) {
  return (
    <NavLink
      to={to}
      end={end}
      onClick={onClick}
      className={({ isActive }) =>
        `group relative flex items-center gap-3 h-10 px-4 mx-2 rounded-md text-sm transition-all duration-200 ${
          isActive
            ? "bg-[#22d1ee]/8 text-[#22d1ee] font-medium"
            : "text-[#848e9c] hover:text-[#eaecef] hover:bg-[#1a2029]"
        }`
      }
    >
      {({ isActive }) => (
        <>
          {isActive && (
            <span className="absolute left-0 top-1/2 -translate-y-1/2 h-5 w-[3px] bg-[#22d1ee] rounded-r-full" />
          )}
          <span className="w-5 h-5 shrink-0 flex items-center justify-center">{icon}</span>
          <span>{label}</span>
        </>
      )}
    </NavLink>
  );
}

export default function Layout() {
  const { connected } = useDashboard();
  const { address, connecting, connect, disconnect } = useWallet();
  const location = useLocation();
  const title = pageTitles[location.pathname] || "XenBlocks";
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const truncAddr = address ? `${address.slice(0, 6)}...${address.slice(-4)}` : "";

  const closeSidebar = () => setSidebarOpen(false);

  return (
    <div className="flex min-h-screen bg-[#0b0e11]">
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/50 md:hidden"
          onClick={closeSidebar}
        />
      )}

      {/* Sidebar */}
      <aside className={`fixed top-0 left-0 bottom-0 w-56 bg-[#141820] border-r border-[#2a3441] flex flex-col z-40 transition-transform duration-300 ease-in-out ${sidebarOpen ? "translate-x-0" : "-translate-x-full"} md:translate-x-0`}>
        {/* Logo */}
        <div className="h-16 px-5 flex items-center gap-3">
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <path
              d="M10 2l6 5-6 11-6-11 6-5z"
              fill="rgba(34,209,238,0.15)"
              stroke="#22d1ee"
              strokeWidth="1.5"
              strokeLinejoin="round"
            />
          </svg>
          <span className="text-[15px] font-bold tracking-tight text-[#eaecef]">
            XenBlocks
          </span>
          {/* Mobile close button */}
          <button
            onClick={closeSidebar}
            className="ml-auto md:hidden text-[#848e9c] hover:text-[#eaecef] transition-colors"
          >
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M5 5l10 10M15 5L5 15" />
            </svg>
          </button>
        </div>
        <div className="h-px bg-[#1f2835] mx-4" />

        {/* Nav — Public */}
        <nav className="flex-1 mt-4 flex flex-col overflow-y-auto">
          <div className="space-y-1">
            {publicNav.map((item) => (
              <SideNavLink key={item.to} {...item} end={item.to === "/"} onClick={closeSidebar} />
            ))}
          </div>

          {/* Separator */}
          <div className="mx-4 my-3 flex items-center gap-2">
            <div className="flex-1 h-px bg-[#1f2835]" />
            <span className="text-[10px] uppercase tracking-widest text-[#5e6673] select-none">Wallet</span>
            <div className="flex-1 h-px bg-[#1f2835]" />
          </div>

          {/* Nav — Wallet */}
          <div className="space-y-1">
            {walletNav.map((item) => (
              <SideNavLink key={item.to} {...item} onClick={closeSidebar} />
            ))}
          </div>
        </nav>

        {/* Bottom */}
        <div className="relative">
          <div className="h-12 bg-gradient-to-t from-[#141820] to-transparent pointer-events-none" />
          <div className="px-5 pb-4 flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs">
              <span
                className={`w-2 h-2 rounded-full ${
                  connected
                    ? "bg-[#0ecb81] shadow-[0_0_6px_rgba(14,203,129,0.5)]"
                    : "bg-[#f6465d]"
                }`}
              />
              <span className={connected ? "text-[#0ecb81]" : "text-[#f6465d]"}>
                {connected ? "Connected" : "Disconnected"}
              </span>
            </div>
            <span className="text-xs text-[#5e6673]">v0.3.0</span>
          </div>
        </div>
      </aside>

      {/* Main */}
      <div className="md:ml-56 flex-1 flex flex-col min-h-screen bg-[#0b0e11]">
        {/* Top bar */}
        <header className="sticky top-0 z-10 h-14 bg-[#141820]/80 backdrop-blur-md border-b border-[#2a3441] flex items-center justify-between px-4 sm:px-6 shrink-0">
          <div className="flex items-center gap-3">
            {/* Hamburger — mobile only */}
            <button
              onClick={() => setSidebarOpen(true)}
              className="md:hidden text-[#848e9c] hover:text-[#eaecef] transition-colors -ml-1"
            >
              <svg width="22" height="22" viewBox="0 0 22 22" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                <path d="M4 6h14M4 11h14M4 16h14" />
              </svg>
            </button>
            <h1 className="text-sm font-semibold text-[#eaecef]">{title}</h1>
          </div>
          <div className="flex items-center gap-2 sm:gap-4">
            <input
              type="text"
              placeholder="Search..."
              className="hidden sm:block bg-[#0b0e11] border border-[#2a3441] rounded-md px-3 py-1.5 text-xs text-[#eaecef] placeholder-[#5e6673] focus:border-[#22d1ee] focus:outline-none w-48 transition-colors"
            />
            {address ? (
              <button
                onClick={disconnect}
                className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-[#1f2835] border border-[#2a3441] text-xs font-mono text-[#22d1ee] hover:border-[#22d1ee] hover:shadow-[0_0_12px_rgba(34,209,238,0.15)] transition-all duration-200"
              >
                <span className="w-2 h-2 rounded-full bg-[#0ecb81]" />
                {truncAddr}
              </button>
            ) : (
              <button
                onClick={connect}
                disabled={connecting}
                className="px-3 py-1.5 rounded-md bg-[#22d1ee]/10 border border-[#22d1ee]/30 text-xs font-medium text-[#22d1ee] hover:bg-[#22d1ee]/20 hover:shadow-[0_0_12px_rgba(34,209,238,0.15)] transition-all duration-200 disabled:opacity-50 whitespace-nowrap"
              >
                {connecting ? "Connecting..." : "Connect Wallet"}
              </button>
            )}
            <button className="text-[#848e9c] hover:text-[#eaecef] transition-colors">
              <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M13.5 6.75a4.5 4.5 0 10-9 0c0 5.25-2.25 6.75-2.25 6.75h13.5s-2.25-1.5-2.25-6.75" />
                <path d="M10.3 15.75a1.5 1.5 0 01-2.6 0" />
              </svg>
            </button>
          </div>
        </header>

        {/* Content */}
        <main className="flex-1 overflow-y-auto p-4 sm:p-6">
          <div key={location.pathname} className="animate-page-enter">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}
