import { useEffect, useRef, useState, useCallback } from "react";
import type { Worker, Block, Stats, WSMessage } from "../types";

const OFFLINE_THRESHOLD = 90;

interface DashboardState {
  workers: Worker[];
  stats: Stats;
  recentBlocks: Block[];
  connected: boolean;
}

const defaultStats: Stats = {
  total_workers: 0,
  online: 0,
  offline: 0,
  total_hashrate: 0,
  total_gpus: 0,
  active_gpus: 0,
  total_blocks: 0,
  blocks_last_hour: 0,
};

export function useWebSocket(): DashboardState {
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [stats, setStats] = useState<Stats>(defaultStats);
  const [recentBlocks, setRecentBlocks] = useState<Block[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const retryRef = useRef(1000);

  const connect = useCallback(() => {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${proto}//${location.host}/ws/dashboard`);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      retryRef.current = 1000;
    };

    ws.onclose = () => {
      setConnected(false);
      const delay = retryRef.current;
      retryRef.current = Math.min(delay * 2, 30000);
      setTimeout(connect, delay);
    };

    ws.onerror = () => ws.close();

    ws.onmessage = (ev) => {
      try {
        const msg: WSMessage = JSON.parse(ev.data);
        handleMessage(msg);
      } catch {
        // ignore malformed messages
      }
    };
  }, []);

  const handleMessage = useCallback((msg: WSMessage) => {
    switch (msg.type) {
      case "snapshot": {
        const d = msg.data;
        const now = Date.now() / 1000;
        const ws: Worker[] = (d.workers || []).map((w: Worker) => ({
          ...w,
          online: now - w.last_heartbeat < OFFLINE_THRESHOLD,
        }));
        setWorkers(ws);
        setRecentBlocks(d.recent_blocks || []);
        recomputeStats(ws);
        break;
      }
      case "heartbeat": {
        const d = msg.data;
        setWorkers((prev) => {
          const idx = prev.findIndex((w) => w.worker_id === d.worker_id);
          if (idx === -1) return prev;
          const updated = [...prev];
          updated[idx] = {
            ...updated[idx],
            hashrate: d.hashrate,
            active_gpus: d.active_gpus,
            last_heartbeat: msg.ts,
            online: true,
          };
          recomputeStats(updated);
          return updated;
        });
        break;
      }
      case "block": {
        const d = msg.data;
        const block: Block = {
          lease_id: d.lease_id || "",
          worker_id: d.worker_id || "",
          hash: d.hash || "",
          key: d.key || "",
          account: d.account || "",
          attempts: d.attempts || 0,
          hashrate: d.hashrate || "0",
          prefix_valid: d.prefix_valid ?? true,
          chain_verified: d.chain_verified ?? false,
          chain_block_id: d.chain_block_id ?? null,
          timestamp: msg.ts,
        };
        setRecentBlocks((prev) => [block, ...prev].slice(0, 20));
        // Bump block count in stats
        setStats((prev) => ({
          ...prev,
          total_blocks: prev.total_blocks + 1,
          blocks_last_hour: prev.blocks_last_hour + 1,
        }));
        break;
      }
      case "health": {
        const list: { worker_id: string }[] = Array.isArray(msg.data)
          ? msg.data
          : [];
        const offlineIds = new Set(list.map((h) => h.worker_id));
        setWorkers((prev) => {
          const updated = prev.map((w) =>
            offlineIds.has(w.worker_id) ? { ...w, online: false } : w,
          );
          recomputeStats(updated);
          return updated;
        });
        break;
      }
    }
  }, []);

  const recomputeStats = useCallback((ws: Worker[]) => {
    let totalHashrate = 0;
    let onlineCount = 0;
    let offlineCount = 0;
    let totalGpus = 0;
    let activeGpus = 0;
    for (const w of ws) {
      totalGpus += w.gpu_count;
      if (w.online) {
        onlineCount++;
        totalHashrate += w.hashrate;
        activeGpus += w.active_gpus;
      } else {
        offlineCount++;
      }
    }
    setStats((prev) => ({
      ...prev,
      total_workers: ws.length,
      online: onlineCount,
      offline: offlineCount,
      total_hashrate: totalHashrate,
      total_gpus: totalGpus,
      active_gpus: activeGpus,
    }));
  }, []);

  useEffect(() => {
    // Fetch initial stats (blocks count) from REST
    fetch("/api/monitoring/stats")
      .then((r) => r.json())
      .then((s: Stats) => setStats((prev) => ({ ...prev, total_blocks: s.total_blocks, blocks_last_hour: s.blocks_last_hour })))
      .catch(() => {});
    connect();
    return () => wsRef.current?.close();
  }, [connect]);

  return { workers, stats, recentBlocks, connected };
}
