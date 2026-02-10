import { createContext, useContext } from "react";
import type { Worker, Block, Stats } from "../types";
import { useWebSocket } from "../hooks/useWebSocket";

interface DashboardCtx {
  workers: Worker[];
  stats: Stats;
  recentBlocks: Block[];
  connected: boolean;
}

const Ctx = createContext<DashboardCtx>({
  workers: [],
  stats: {
    total_workers: 0, online: 0, offline: 0,
    total_hashrate: 0, total_gpus: 0, active_gpus: 0,
    total_blocks: 0, blocks_last_hour: 0,
  },
  recentBlocks: [],
  connected: false,
});

export function DashboardProvider({ children }: { children: React.ReactNode }) {
  const value = useWebSocket();
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useDashboard() {
  return useContext(Ctx);
}
