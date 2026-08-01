import React, { Suspense, useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';

// FIX BUG-002: ErrorBoundary must be eagerly imported — never lazy loaded
import ErrorBoundary from './components/ErrorBoundary';
import { DiagnosisProvider } from './context/DiagnosisContext';
import DisclaimerModal from './components/DisclaimerModal';
import DisclaimerGuard from './components/DisclaimerGuard';

import { ConsentModal } from './components/ConsentModal';

const Landing   = React.lazy(() => import('./pages/Landing'));
const Login     = React.lazy(() => import('./pages/Login'));
const Dashboard = React.lazy(() => import('./pages/Dashboard'));
const Diagnosis = React.lazy(() => import('./pages/Diagnosis'));
const History   = React.lazy(() => import('./pages/History'));
const Result    = React.lazy(() => import('./pages/Result'));
const Layout    = React.lazy(() => import('./components/Layout'));
const ResetPassword = React.lazy(() => import('./pages/ResetPassword'));

const ProtectedRoute = ({ children }) => {
  const token = localStorage.getItem('token');
  if (!token) return <Navigate to="/login" replace />;
  return children;
};

function App() {
  const [consentGiven, setConsentGiven] = useState(true);

  useEffect(() => {
    const given = sessionStorage.getItem("tb_consent");
    if (!given) setConsentGiven(false);
  }, []);

  const handleAccept = () => {
    sessionStorage.setItem("tb_consent", "true");
    setConsentGiven(true);
  };

  if (!consentGiven) {
    return <ConsentModal onAccept={handleAccept} />;
  }

  return (
    <Router>
      <div className="relative min-h-screen bg-[#f8fafc] text-slate-800 font-sans selection:bg-blue-200">
        <ErrorBoundary>
          <DiagnosisProvider>
            <Suspense fallback={
              <div className="flex items-center justify-center h-screen bg-[#f8fafc]">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600" />
              </div>
            }>
              <Routes>
                <Route path="/" element={<Landing />} />
                <Route path="/login" element={<Login />} />
                <Route path="/reset-password" element={<ResetPassword />} />

                {/* Authenticated shell at /app/* */}
                <Route path="/app" element={
                  <ProtectedRoute>
                    <Layout />
                  </ProtectedRoute>
                }>
                  <Route index element={<Navigate to="dashboard" replace />} />
                  <Route path="dashboard" element={<Dashboard />} />
                  <Route path="diagnosis" element={<Diagnosis />} />

                  {/* BUG C-01: Results page has a second guard layer —
                      if someone manually clears localStorage mid-session,
                      DisclaimerGuard catches it and redirects to / */}
                  <Route path="result" element={
                    <DisclaimerGuard>
                      <Result />
                    </DisclaimerGuard>
                  } />

                  <Route path="history" element={<History />} />
                </Route>

                {/* Shorthand redirects */}
                <Route path="/dashboard" element={<Navigate to="/app/dashboard" replace />} />
                <Route path="/diagnosis" element={<Navigate to="/app/diagnosis" replace />} />
                <Route path="/history"   element={<Navigate to="/app/history"   replace />} />

                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </Suspense>
          </DiagnosisProvider>
        </ErrorBoundary>
      </div>

      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: '#ffffff',
            color: '#1e293b',
            border: '1px solid #e2e8f0',
            boxShadow: '0 4px 15px rgba(0,0,0,0.05)',
            fontWeight: 600,
            fontSize: '14px',
          },
          success: { iconTheme: { primary: '#10b981', secondary: '#ffffff' } },
          error:   { iconTheme: { primary: '#ef4444', secondary: '#ffffff' } },
        }}
      />
    </Router>
  );
}

export default App;
