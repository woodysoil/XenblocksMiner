import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from "react";
import { BrowserProvider } from "ethers";
import { getToken, setToken, clearToken, apiFetch } from "../api";

interface WalletCtx {
  address: string | null;
  connecting: boolean;
  connect: () => Promise<void>;
  disconnect: () => void;
}

const Ctx = createContext<WalletCtx>({
  address: null,
  connecting: false,
  connect: async () => {},
  disconnect: () => {},
});

export function WalletProvider({ children }: { children: ReactNode }) {
  const [address, setAddress] = useState<string | null>(null);
  const [connecting, setConnecting] = useState(false);

  // Restore session on mount
  useEffect(() => {
    const token = getToken();
    if (!token) return;
    apiFetch<{ eth_address: string }>("/api/auth/me")
      .then((me) => {
        if (me.eth_address) setAddress(me.eth_address);
        else { clearToken(); }
      })
      .catch(() => { clearToken(); });
  }, []);

  // Listen for MetaMask account/chain changes
  useEffect(() => {
    const eth = (window as any).ethereum;
    if (!eth) return;
    const handleChange = () => { clearToken(); setAddress(null); };
    eth.on("accountsChanged", handleChange);
    eth.on("chainChanged", handleChange);
    return () => {
      eth.removeListener("accountsChanged", handleChange);
      eth.removeListener("chainChanged", handleChange);
    };
  }, []);

  const connect = useCallback(async () => {
    const eth = (window as any).ethereum;
    if (!eth) {
      alert("MetaMask not detected");
      return;
    }
    setConnecting(true);
    try {
      const provider = new BrowserProvider(eth);
      const signer = await provider.getSigner();
      const addr = await signer.getAddress();

      const { message, nonce } = await apiFetch<{ nonce: string; message: string }>(
        `/api/auth/nonce?address=${addr}`,
      );

      const signature = await signer.signMessage(message);

      const result = await apiFetch<{ token: string; address: string }>("/api/auth/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ address: addr, signature, nonce }),
      });

      setToken(result.token);
      setAddress(result.address);
    } catch (e: any) {
      if (e?.code !== 4001) console.error("Wallet connect failed:", e);
    } finally {
      setConnecting(false);
    }
  }, []);

  const disconnect = useCallback(() => {
    clearToken();
    setAddress(null);
  }, []);

  return (
    <Ctx.Provider value={{ address, connecting, connect, disconnect }}>
      {children}
    </Ctx.Provider>
  );
}

export function useWallet() {
  return useContext(Ctx);
}
