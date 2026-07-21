import React from 'react';
import { Outlet, NavLink, useNavigate } from 'react-router-dom';
import { LayoutDashboard, FileScan, LogOut, User } from 'lucide-react';
import { toast } from 'react-hot-toast';

export default function Layout() {
  const navigate = useNavigate();
  const username = localStorage.getItem('username') || 'Doctor';

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('username');
    toast('Logged out successfully', { icon: '👋' });
    navigate('/login');
  };

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 glass-panel m-4 flex flex-col justify-between hidden md:flex">
        <div>
          <div className="p-6 border-b border-white/10 flex items-center gap-3 text-primary">
            <div className="h-4 w-4 rounded-full bg-primary animate-pulse"></div>
            <h1 className="text-xl font-bold text-white tracking-wide">TB<span className="text-primary">-</span>Vision</h1>
          </div>
          
          <nav className="p-4 space-y-2 mt-4">
            <NavLink 
              to="/dashboard" 
              className={({ isActive }) => `flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${isActive ? 'bg-primary/20 text-primary font-semibold' : 'text-textmuted hover:bg-white/5 hover:text-white'}`}
            >
              <LayoutDashboard size={20} />
              <span>Dashboard</span>
            </NavLink>
            <NavLink 
              to="/diagnosis" 
              className={({ isActive }) => `flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${isActive ? 'bg-primary/20 text-primary font-semibold' : 'text-textmuted hover:bg-white/5 hover:text-white'}`}
            >
              <FileScan size={20} />
              <span>New Diagnosis</span>
            </NavLink>
          </nav>
        </div>

        <div className="p-4 border-t border-white/10">
          <div className="flex items-center gap-3 px-4 py-3 mb-2">
            <div className="rounded-full bg-secondary/20 p-2 text-secondary">
              <User size={20} />
            </div>
            <div>
              <p className="text-sm font-semibold text-white truncate w-32">{username}</p>
              <p className="text-xs text-textmuted">Medical Professional</p>
            </div>
          </div>
          <button 
            onClick={handleLogout}
            className="flex items-center gap-3 px-4 py-3 w-full text-left rounded-xl transition-all text-red-400 hover:bg-red-400/10"
          >
            <LogOut size={20} />
            <span>Logout</span>
          </button>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 overflow-y-auto p-4 md:p-8">
        {/* Mobile Header */}
        <div className="md:hidden glass-panel p-4 mb-4 flex justify-between items-center rounded-2xl">
           <h1 className="text-xl font-bold text-white tracking-wide">TB<span className="text-primary">-</span>Vision</h1>
           <button onClick={handleLogout} className="text-textmuted hover:text-red-400"><LogOut size={20}/></button>
        </div>
        <Outlet />
      </main>
    </div>
  );
}
