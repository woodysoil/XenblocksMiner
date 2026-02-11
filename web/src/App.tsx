import { Suspense, lazy } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "sonner";
import { queryClient } from "./lib/queryClient";
import { DashboardProvider } from "./context/DashboardContext";
import { WalletProvider } from "./context/WalletContext";
import Layout from "./components/Layout";
import ErrorBoundary from "./components/ErrorBoundary";

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
            <ErrorBoundary>
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
                    <Route index element={<ErrorBoundary><Overview /></ErrorBoundary>} />
                    <Route path="monitoring" element={<ErrorBoundary><Monitoring /></ErrorBoundary>} />
                    <Route path="marketplace" element={<ErrorBoundary><Marketplace /></ErrorBoundary>} />
                    <Route path="provider" element={<ErrorBoundary><Provider /></ErrorBoundary>} />
                    <Route path="renter" element={<ErrorBoundary><Renter /></ErrorBoundary>} />
                    <Route path="account" element={<ErrorBoundary><Account /></ErrorBoundary>} />
                    <Route path="*" element={<NotFound />} />
                  </Route>
                </Routes>
              </Suspense>
            </ErrorBoundary>
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
