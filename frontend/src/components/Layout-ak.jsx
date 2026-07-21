import React from 'react';
import { Outlet, NavLink, useNavigate, useLocation } from 'react-router-dom';
import { Home, Activity, LogOut, Clock, User, Zap } from 'lucide-react';
import { toast } from 'react-hot-toast';
import { motion, AnimatePresence } from 'framer-motion';
import PatientDataNotice from './PatientDataNotice';
import { logConsent } from '../services/api';
import { useDiagnosis } from '../context/DiagnosisContext';

export default function Layout() {
  const navigate = useNavigate();
  const location = useLocation();
  const { state, dispatch } = useDiagnosis();
  const username = localStorage.getItem('username') || 'Doctor';

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('username');
    toast.success('Session terminated securely');
    navigate('/login');
  };

  const [showConsentModal, setShowConsentModal] = React.useState(false);
  const hasConsented = state.hasConsented;

  const handleAcceptConsent = async () => {
    try {
      const sessionId = `sess_${Math.random().toString(36).substring(2, 15)}`;
      await logConsent(sessionId);
      sessionStorage.setItem('tb_consent_ack', 'true');
      dispatch({ type: 'SET_CONSENT', payload: true });
      setShowConsentModal(false);
      toast.success('Privacy & Consent policy acknowledged');
    } catch (err) {
      toast.error('Audit sync failed. Please try again.');
    }
  };

  const navItems = [
    { to: '/app/dashboard', icon: Home,     label: 'Dashboard'   },
    { to: '/app/diagnosis',  icon: Activity, label: 'Diagnostics' },
    { to: '/app/history',    icon: Clock,    label: 'Archives'    },
  ];

  return (
    <div className="flex h-screen bg-[#f8fafc] overflow-hidden font-sans">

      {/* Sidebar — Desktop */}
      <aside className="w-[100px] hidden lg:flex flex-col items-center py-10 border-r border-slate-200/50 bg-slate-50/50 relative z-50">
        <div className="w-12 h-12 bg-blue-600 rounded-2xl mb-12 flex items-center justify-center text-white font-black text-xl shadow-[0_10px_30px_rgba(37,99,235,0.4)]">
          TB
        </div>

        <nav className="flex flex-col gap-10">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `p-3.5 rounded-2xl transition-all relative group ${
                  isActive
                    ? 'text-blue-600 bg-white shadow-xl shadow-slate-200/50 border border-slate-100'
                    : 'text-slate-400 hover:text-blue-600 hover:bg-white'
                }`
              }
            >
              <item.icon size={26} />
              <span className="absolute left-full ml-4 px-2 py-1 bg-slate-900 text-white text-[10px] font-black uppercase tracking-widest rounded opacity-0 group-hover:opacity-100 pointer-events-none transition-all whitespace-nowrap z-[100]">
                {item.label}
              </span>
            </NavLink>
          ))}
        </nav>

        <div className="mt-auto flex flex-col gap-8 items-center">
          <div className="p-1.5 bg-white rounded-2xl border border-slate-100 shadow-sm cursor-pointer hover:border-blue-200 transition-all" title={username}>
            <div className="w-9 h-9 rounded-xl bg-slate-100 flex items-center justify-center text-slate-400">
              <User size={20} />
            </div>
          </div>
          <button
            onClick={handleLogout}
            className="p-3.5 text-slate-400 hover:text-rose-500 hover:bg-rose-50 rounded-2xl transition-all active:scale-95"
            title="Logout"
          >
            <LogOut size={26} />
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col h-full overflow-hidden relative">

        {/* Mobile Header */}
        <div className="lg:hidden bg-white/80 backdrop-blur-xl p-5 border-b border-slate-200 flex justify-between items-center z-50">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center text-white font-black">TB</div>
            <h1 className="text-xl font-black text-slate-900 tracking-tight font-[Poppins]">Vision Pro</h1>
          </div>
          <button onClick={handleLogout} className="p-2.5 bg-slate-50 rounded-xl text-slate-400 hover:text-rose-500" title="Logout">
            <LogOut size={22} />
          </button>
        </div>

        {/* FIX BUG-017: pb-28 on mobile so content isn't behind bottom nav */}
        <div className="flex-1 overflow-y-auto overflow-x-hidden custom-scrollbar pb-28 lg:pb-0">
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 10, scale: 0.99 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -10, scale: 0.99 }}
              transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
              className="w-full h-full"
            >
              <Outlet />
            </motion.div>
          </AnimatePresence>
        </div>



        {/* Global Patient Data Notice */}
        <AnimatePresence>
           {(showConsentModal || (!hasConsented && location.pathname === '/app/diagnosis')) && (
              <PatientDataNotice 
                 onAccept={handleAcceptConsent} 
                 onCancel={() => location.pathname === '/app/diagnosis' ? navigate('/app/dashboard') : setShowConsentModal(false)} 
              />
           )}
        </AnimatePresence>

        {/* Persistent Footer Link - Hidden on mobile to make room for Nav */}
        <div className="absolute bottom-6 left-0 right-0 hidden lg:flex justify-center pointer-events-none z-[101]">
           <button 
              onClick={() => setShowConsentModal(true)}
              className="pointer-events-auto px-4 py-2 bg-white/50 backdrop-blur-md rounded-full text-[9px] font-black uppercase tracking-[0.2em] text-slate-400 hover:text-blue-600 hover:bg-white transition-all shadow-sm border border-slate-200/50"
           >
              Privacy Policy & Data Processing Notice
           </button>
        </div>

        {/* Mobile Bottom Navigation Bar */}
        <nav className="lg:hidden fixed bottom-0 left-0 right-0 bg-white/90 backdrop-blur-xl border-t border-slate-100 flex items-center justify-around p-3 z-50 pb-8">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `flex flex-col items-center gap-1 p-2 transition-all ${
                  isActive ? 'text-blue-600' : 'text-slate-400'
                }`
              }
            >
              <item.icon size={20} />
              <span className="text-[10px] font-black uppercase tracking-widest">{item.label}</span>
              {location.pathname === item.to && (
                <motion.div layoutId="mobileNavTab" className="w-1 h-1 bg-blue-600 rounded-full" />
              )}
            </NavLink>
          ))}
        </nav>
      </main>
    </div>
  );
}
