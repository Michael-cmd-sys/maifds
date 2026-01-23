import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { DashboardLayout } from '@/components/layout/DashboardLayout';
import Overview from '@/pages/Overview';
import Playground from '@/pages/Playground';
import Blacklist from '@/pages/Blacklist';
import Reputation from '@/pages/Reputation';
import Privacy from '@/pages/Privacy';
import Audit from '@/pages/Audit';
import Alerts from '@/pages/Alerts';
import { Toaster } from "@/components/ui/toaster"

function App() {
  return (
    <BrowserRouter basename="/app">
      <Routes>
        <Route path="/" element={<DashboardLayout />}>
          <Route index element={<Navigate to="/overview" replace />} />
          <Route path="overview" element={<Overview />} />
          <Route path="playground" element={<Playground />} />
          <Route path="blacklist" element={<Blacklist />} />
          <Route path="reputation" element={<Reputation />} />
          <Route path="governance/privacy" element={<Privacy />} />
          <Route path="governance/audit" element={<Audit />} />
          <Route path="alerts" element={<Alerts />} />
        </Route>
      </Routes>
      <Toaster />
    </BrowserRouter>
  );
}

export default App;
