import { BrowserRouter, Routes, Route } from "react-router-dom";
import { DashboardProvider } from "./context/DashboardContext";
import { WalletProvider } from "./context/WalletContext";
import Layout from "./components/Layout";
import Overview from "./pages/Overview";
import Monitoring from "./pages/Monitoring";
import Marketplace from "./pages/Marketplace";
import Provider from "./pages/Provider";

export default function App() {
  return (
    <BrowserRouter>
      <WalletProvider>
        <DashboardProvider>
          <Routes>
            <Route element={<Layout />}>
              <Route index element={<Overview />} />
              <Route path="monitoring" element={<Monitoring />} />
              <Route path="marketplace" element={<Marketplace />} />
              <Route path="provider" element={<Provider />} />
            </Route>
          </Routes>
        </DashboardProvider>
      </WalletProvider>
    </BrowserRouter>
  );
}
