export function formatHashrate(h: number): string {
  if (h >= 1e9) return (h / 1e9).toFixed(2) + " GH/s";
  if (h >= 1e6) return (h / 1e6).toFixed(2) + " MH/s";
  if (h >= 1e3) return (h / 1e3).toFixed(2) + " KH/s";
  return h.toFixed(1) + " H/s";
}

export function formatHashrateCompact(h: number): string {
  if (h >= 1e6) return (h / 1e6).toFixed(2) + " MH";
  if (h >= 1e3) return (h / 1e3).toFixed(2) + " KH";
  return h.toFixed(1);
}

export function timeAgo(ts: number | string): { text: string; sec: number } {
  const ms = typeof ts === "number" ? (ts > 1e12 ? ts : ts * 1000) : new Date(ts).getTime();
  const sec = Math.max(0, Math.floor((Date.now() - ms) / 1000));
  let text: string;
  if (sec < 60) text = `${sec}s ago`;
  else if (sec < 3600) text = `${Math.floor(sec / 60)}m ago`;
  else if (sec < 86400) text = `${Math.floor(sec / 3600)}h ago`;
  else text = `${Math.floor(sec / 86400)}d ago`;
  return { text, sec };
}

export function formatUptime(sec: number): string {
  const d = Math.floor(sec / 86400);
  const h = Math.floor((sec % 86400) / 3600);
  if (d > 0) return `${d}d ${h}h`;
  if (h > 0) return `${h}h`;
  return `${Math.floor(sec / 60)}m`;
}
