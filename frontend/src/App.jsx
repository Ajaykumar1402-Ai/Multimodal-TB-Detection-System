import React, { Suspense } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import BackgroundScene from './components/3D/BackgroundScene';

// Lazy loaded pages to optimize initial bundle
const Login = React.lazy(() => import('./pages/Login'));
const Dashboard = React.lazy(() => import('./pages/Dashboard'));
const Diagnosis = React.lazy(() => import('./pages/Diagnosis'));
const Layout = React.lazy(() => import('./components/Layout'));

// Auth Guard
const ProtectedRoute = ({ children }) => {
  const token = localStorage.getItem('token');
  if (!token) return <Navigate to="/login" replace />;
  return children;
};

function App() {
  return (
    <Router>
      <div className="relative min-h-screen bg-background text-textmain overflow-hidden">
        {/* Persistent 3D Background */}
        <div className="absolute inset-0 z-0 pointer-events-none">
          <BackgroundScene />
        </div>
        
        {/* Main Content */}
        <div className="relative z-10 min-h-screen">
          <Suspense fallback={<div className="flex items-center justify-center h-screen"><div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div></div>}>
            <Routes>
              <Route path="/login" element={<Login />} />
              <Route path="/" element={
                 <ProtectedRoute>
                    <Layout />
                 </ProtectedRoute>
              }>
                <Route index element={<Navigate to="/dashboard" replace />} />
                <Route path="dashboard" element={<Dashboard />} />
                <Route path="diagnosis" element={<Diagnosis />} />
              </Route>
            </Routes>
          </Suspense>
        </div>
      </div>
      <Toaster position="top-right" theme="dark" toastOptions={{
        style: {
          background: '#1e293b',
          color: '#fff',
          border: '1px solid rgba(255,255,255,0.1)',
        }
      }} />
    </Router>
  );
}

export default App;
