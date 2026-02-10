import { NavLink, Outlet, useLocation } from "react-router-dom";
import { useDashboard } from "../context/DashboardContext";

const navItems = [
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
];

const pageTitles: Record<string, string> = {
  "/": "Overview",
  "/monitoring": "Monitoring",
  "/marketplace": "Marketplace",
  "/provider": "Provider",
};

export default function Layout() {
  const { connected } = useDashboard();
  const location = useLocation();
  const title = pageTitles[location.pathname] || "XenBlocks";

  return (
    <div className="flex min-h-screen bg-[#0b0e11]">
      {/* Sidebar */}
      <aside className="fixed top-0 left-0 bottom-0 w-56 bg-[#141820] border-r border-[#2a3441] flex flex-col z-20">
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
        </div>
        <div className="h-px bg-[#1f2835] mx-4" />

        {/* Nav */}
        <nav className="flex-1 mt-4 space-y-1">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                `flex items-center gap-3 h-10 px-4 mx-2 rounded-md text-sm transition-colors ${
                  isActive
                    ? "bg-[#1f2835] text-[#22d1ee] font-medium"
                    : "text-[#848e9c] hover:text-[#eaecef] hover:bg-[#1a2029]"
                }`
              }
            >
              {item.icon}
              {item.label}
            </NavLink>
          ))}
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
      <div className="ml-56 flex-1 flex flex-col min-h-screen bg-[#0b0e11]">
        {/* Top bar */}
        <header className="h-14 bg-[#141820] border-b border-[#2a3441] flex items-center justify-between px-6 shrink-0">
          <h1 className="text-sm font-semibold text-[#eaecef]">{title}</h1>
          <div className="flex items-center gap-4">
            <input
              type="text"
              placeholder="Search..."
              className="bg-[#0b0e11] border border-[#2a3441] rounded-md px-3 py-1.5 text-xs text-[#eaecef] placeholder-[#5e6673] focus:border-[#22d1ee] focus:outline-none w-48 transition-colors"
            />
            <button className="text-[#848e9c] hover:text-[#eaecef] transition-colors">
              <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M13.5 6.75a4.5 4.5 0 10-9 0c0 5.25-2.25 6.75-2.25 6.75h13.5s-2.25-1.5-2.25-6.75" />
                <path d="M10.3 15.75a1.5 1.5 0 01-2.6 0" />
              </svg>
            </button>
          </div>
        </header>

        {/* Content */}
        <main className="flex-1 p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
