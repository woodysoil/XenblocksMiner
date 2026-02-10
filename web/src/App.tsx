import { BrowserRouter, Routes, Route } from "react-router-dom";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "sonner";
import { queryClient } from "./lib/queryClient";
import { DashboardProvider } from "./context/DashboardContext";
import { WalletProvider } from "./context/WalletContext";
import Layout from "./components/Layout";
import Overview from "./pages/Overview";
import Monitoring from "./pages/Monitoring";
import Marketplace from "./pages/Marketplace";
import Provider from "./pages/Provider";
import Renter from "./pages/Renter";
import Account from "./pages/Account";

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <WalletProvider>
          <DashboardProvider>
            <Routes>
              <Route element={<Layout />}>
                <Route index element={<Overview />} />
                <Route path="monitoring" element={<Monitoring />} />
                <Route path="marketplace" element={<Marketplace />} />
                <Route path="provider" element={<Provider />} />
                <Route path="renter" element={<Renter />} />
                <Route path="account" element={<Account />} />
              </Route>
            </Routes>
          </DashboardProvider>
        </WalletProvider>
      </BrowserRouter>
      <Toaster
        position="bottom-right"
        theme="dark"
        toastOptions={{
          style: {
            background: "#141820",
            border: "1px solid #2a3441",
            color: "#eaecef",
          },
        }}
      />
    </QueryClientProvider>
  );
}
