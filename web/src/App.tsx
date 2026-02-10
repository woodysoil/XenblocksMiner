import { Suspense, lazy } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "sonner";
import { queryClient } from "./lib/queryClient";
import { DashboardProvider } from "./context/DashboardContext";
import { WalletProvider } from "./context/WalletContext";
import Layout from "./components/Layout";

const Overview = lazy(() => import("./pages/Overview"));
const Monitoring = lazy(() => import("./pages/Monitoring"));
const Marketplace = lazy(() => import("./pages/Marketplace"));
const Provider = lazy(() => import("./pages/Provider"));
const Renter = lazy(() => import("./pages/Renter"));
const Account = lazy(() => import("./pages/Account"));
const NotFound = lazy(() => import("./pages/NotFound"));

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <WalletProvider>
          <DashboardProvider>
            <Suspense
              fallback={
                <div className="flex-1 flex items-center justify-center">
                  <div className="animate-pulse text-[#5e6673] text-sm">
                    Loading…
                  </div>
                </div>
              }
            >
              <Routes>
                <Route element={<Layout />}>
                  <Route index element={<Overview />} />
                  <Route path="monitoring" element={<Monitoring />} />
                  <Route path="marketplace" element={<Marketplace />} />
                  <Route path="provider" element={<Provider />} />
                  <Route path="renter" element={<Renter />} />
                  <Route path="account" element={<Account />} />
                  <Route path="*" element={<NotFound />} />
                </Route>
              </Routes>
            </Suspense>
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
