import React, { useState } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { toast } from 'react-hot-toast';
import { KeyRound, ArrowRight, ShieldCheck } from 'lucide-react';
import api from '../services/api';

export default function ResetPassword() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const token = searchParams.get('token');
  
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);

  if (!token) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#0f172a] text-white">
        <div className="text-center">
          <h2 className="text-2xl font-bold mb-4">Invalid Reset Link</h2>
          <p className="text-slate-400 mb-6">No secure token was provided in the URL.</p>
          <button onClick={() => navigate('/login')} className="text-blue-500 hover:underline">
            Return to Login
          </button>
        </div>
      </div>
    );
  }

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (password.length < 6) {
      toast.error('Password must be at least 6 characters');
      return;
    }
    
    setLoading(true);
    try {
      await api.post('/auth/confirm-reset', {
        token,
        new_password: password
      });
      toast.success('Keyphrase completely reset! You can now access your account.');
      navigate('/login');
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Failed to securely reset password.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex bg-[#0f172a] items-center justify-center p-4">
      <div className="w-full max-w-md bg-white rounded-3xl p-10 shadow-2xl relative overflow-hidden">
        <div className="absolute top-0 right-0 p-4 opacity-5 pointer-events-none">
          <ShieldCheck size={200} />
        </div>
        
        <div className="text-center mb-8">
          <div className="w-16 h-16 bg-blue-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
            <KeyRound className="text-blue-600" size={32} />
          </div>
          <h2 className="text-3xl font-black text-slate-900 tracking-tight font-[Poppins]">
            New Password
          </h2>
          <p className="text-slate-500 text-sm mt-2">
            Please enter your new secure keyphrase below.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6 relative z-10">
          <div className="space-y-2">
            <label className="text-[11px] font-black text-slate-400 uppercase tracking-widest px-1 block">
              New Secure Keyphrase
            </label>
            <input 
              type="password" 
              required 
              className="w-full px-5 py-4 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all font-medium text-slate-900 placeholder:text-slate-400"
              placeholder="••••••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>

          <button 
            type="submit" 
            disabled={loading}
            className="w-full py-4.5 bg-blue-600 hover:bg-blue-700 text-white rounded-xl font-black uppercase tracking-widest text-[13px] transition-all flex items-center justify-center gap-3 disabled:opacity-70 group"
          >
            {loading ? 'Securing...' : 'Confirm Reset'}
            {!loading && <ArrowRight size={20} className="group-hover:translate-x-1 transition-transform" />}
          </button>
        </form>
      </div>
    </div>
  );
}
