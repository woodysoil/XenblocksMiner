export interface Worker {
  worker_id: string;
  eth_address: string;
  gpu_count: number;
  total_memory_gb: number;
  gpus: GpuInfo[];
  version: string;
  state: string;
  hashrate: number;
  active_gpus: number;
  last_heartbeat: number;
  registered_at: number;
  price_per_min: number;
  self_blocks_found: number;
  total_online_sec: number;
  last_online_at: number | null;
  online?: boolean;
}

export interface GpuInfo {
  name?: string;
  memory_gb?: number;
}

export interface Block {
  lease_id: string;
  worker_id: string;
  hash: string;
  key: string;
  account: string;
  attempts: number;
  hashrate: string;
  prefix_valid: boolean;
  chain_verified: boolean;
  chain_block_id: number | null;
  timestamp: number;
}

export interface Stats {
  total_workers: number;
  online: number;
  offline: number;
  total_hashrate: number;
  total_gpus: number;
  active_gpus: number;
  total_blocks: number;
  blocks_last_hour: number;
}

export interface HashratePoint {
  worker_id: string;
  hashrate: number;
  active_gpus: number;
  timestamp: number;
}

export interface WSMessage {
  type: "snapshot" | "heartbeat" | "block" | "health";
  data: any;
  ts: number;
}

export interface WalletSnapshot {
  timestamp: number;
  hashrate: number;
  online_workers: number;
  total_workers: number;
  blocks: number;
  cumulative_blocks: number;
  earnings: number;
}

export interface WalletAchievements {
  address: string;
  current_hashrate: number;
  online_workers: number;
  total_workers: number;
  total_blocks: number;
  peak_hashrate: number;
  total_earnings: number;
  mining_days: number;
  first_seen: number | null;
}

export interface WalletStats {
  address: string;
  online_workers: number;
  total_workers: number;
  total_hashrate: number;
  total_gpus: number;
  total_blocks: number;
  total_earnings: number;
}
