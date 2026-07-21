import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import api, { login, register, requestPasswordReset } from '../services/api';
import { toast } from 'react-hot-toast';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ShieldCheck, Mail, Lock, User,
  ArrowRight, HeartPulse, Activity, Zap,
  KeyRound, UserPlus, LogIn
} from 'lucide-react';

export default function Login() {
  const [mode, setMode] = useState('login'); // 'login' | 'signup' | 'reset'
  const [loading, setLoading] = useState(false);
  const [resetSent, setResetSent] = useState(false);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    full_name: '',
  });
  const [error, setError] = useState('');
  const [status, setStatus] = useState('');
  const navigate = useNavigate();
  
  const [retryStatus, setRetryStatus] = useState('');
  const [resendCountdown, setResendCountdown] = useState(0);

  useEffect(() => {
    // Backend wake-up is now handled by the primary login request to prevent pool exhaustion
    const handleRetry = (e) => {
      setRetryStatus(`(Retry ${e.detail.retryCount}/${e.detail.maxRetries})`);
    };
    window.addEventListener('api-retry', handleRetry);
    return () => window.removeEventListener('api-retry', handleRetry);
  }, []);

  useEffect(() => {
    let timer;
    if (resendCountdown > 0) {
      timer = setInterval(() => {
        setResendCountdown(prev => prev - 1);
      }, 1000);
    }
    return () => clearInterval(timer);
  }, [resendCountdown]);

  const handleAutoRetry = () => {
    setError('Retrying automatically in 30 seconds...');
    setTimeout(() => {
      // Trigger login submit with active credentials
      const mockEvent = { preventDefault: () => {} };
      handleSubmit(mockEvent);
    }, 30000);
  };

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    if (mode === 'reset' && resendCountdown > 0) return;

    setError('');
    setLoading(true);
    setRetryStatus('');
    setStatus(mode === 'login' ? 'Connecting to clinical server...' 
             : mode === 'signup' ? 'Creating medical credentials...' 
             : 'Submitting secure recovery request...');

    try {
      if (mode === 'login') {
        const data = await login(formData.email, formData.password);
        setStatus('Login successful — loading dashboard...');
        localStorage.setItem('token', data.access_token);
        localStorage.setItem('username', data.user_name || data.username);
        toast.success(`Welcome back, Dr. ${data.user_name || data.username}!`);
        navigate('/app/dashboard');
      } else if (mode === 'signup') {
        await register(formData.email, formData.password, formData.full_name);
        toast.success('Medical account created. Please login.');
        setMode('login');
      } else if (mode === 'reset') {
        await requestPasswordReset(formData.email);
        setResetSent(true);
        setResendCountdown(60);
      }
    } catch (err) {
      console.error('LOGIN ERROR:', err);
      const status_code = err.response?.status;
      const detail_msg = err.response?.data?.detail || err.message || '';

      // Map every error to a clear human message
      const errorMap = {
        400: 'Please enter both email and password.',
        401: detail_msg.toLowerCase().includes('password')
               ? 'Incorrect password. Please try again.'
               : 'No account found with this email address.',
        403: 'Your account does not have access. Contact your administrator.',
        429: 'Too many login attempts. Please wait 5 minutes and try again.',
        500: 'Server error. The backend may be starting up — please wait 30 seconds and try again.',
        503: 'Server is starting up after inactivity. Please wait 30 seconds and try again.',
      };

      let resolvedError = errorMap[status_code] || detail_msg || 'Login failed. Please try again.';

      if (err.code === 'ECONNABORTED' || err.message?.includes('timeout') || err.message?.includes('Network Error')) {
        resolvedError = 'Server is waking up from sleep mode. This happens after periods of inactivity. Please wait 30 seconds and try again.';
      }

      setError(resolvedError);
      toast.error(resolvedError);
    } finally {
      setLoading(false);
      setRetryStatus('');
      setStatus('');
    }
  };

  const titles = {
    login:  { heading: 'Welcome Back',       sub: 'Clinical Authentication',     icon: <LogIn size={20} />    },
    signup: { heading: 'Initialize Profile', sub: 'Join the diagnostic network', icon: <UserPlus size={20} /> },
    reset:  { heading: 'Reset Access',       sub: 'Secure clinical recovery',    icon: <KeyRound size={20} /> },
  };

  return (
    <div className="min-h-screen flex bg-[#0f172a] overflow-hidden font-sans">

      {/* Left Panel */}
      <div className="hidden lg:flex w-[55%] relative overflow-hidden bg-slate-950 border-r border-white/5">
        <div className="absolute inset-0 z-0">
          <div className="absolute top-[-10%] left-[-10%] w-[60%] h-[60%] bg-blue-600/20 blur-[130px] rounded-full animate-pulse" />
          <div className="absolute bottom-[-10%] right-[-10%] w-[60%] h-[60%] bg-cyan-600/20 blur-[130px] rounded-full animate-pulse delay-700" />
        </div>

        <div className="relative z-10 w-full h-full flex flex-col p-20 justify-between">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-blue-600 rounded-2xl shadow-[0_0_40px_rgba(37,99,235,0.5)]">
              <ShieldCheck className="text-white" size={32} />
            </div>
            <h1 className="text-3xl font-black text-white tracking-tighter font-[Poppins]">
              TB-Vision <span className="text-blue-500">Pro</span>
            </h1>
          </div>

          <div className="max-w-2xl">
            <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="mb-10">
              <span className="px-5 py-2.5 bg-blue-500/10 border border-blue-500/20 text-blue-400 text-[10px] font-black rounded-full tracking-[0.3em] mb-8 inline-block uppercase">
                Medical Intelligence System v2.0
              </span>
              <h2 className="text-7xl font-black text-white leading-[1] tracking-tight mb-10 font-[Poppins]">
                Empowering <br /><span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-blue-200 to-cyan-300">Precision</span> Care.
              </h2>
              <p className="text-xl text-slate-400 leading-relaxed font-medium">
                Leverage clinical-grade AI and multimodal data to detect Tuberculosis with unparalleled accuracy.
              </p>
            </motion.div>

            <div className="flex gap-12 mt-16 pr-20">
              <div className="flex-1">
                <p className="text-4xl font-black text-white">96.7%</p>
                <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest mt-2">Model Accuracy</p>
              </div>
              <div className="w-px h-16 bg-white/10" />
              <div className="flex-1">
                <p className="text-4xl font-black text-white">0.3s</p>
                <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest mt-2">Inference Speed</p>
              </div>
              <div className="w-px h-16 bg-white/10" />
              <div className="flex-1">
                <p className="text-4xl font-black text-white">Cloud</p>
                <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest mt-2">Synchronized</p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-5 text-slate-500 text-sm font-bold">
            <div className="flex -space-x-4">
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="w-11 h-11 rounded-full border-2 border-slate-900 bg-slate-800 flex items-center justify-center">
                  <User size={18} className="text-slate-400" />
                </div>
              ))}
            </div>
            <p>Trusted by <span className="text-blue-400">1,200+</span> Clinical Professionals</p>
          </div>
        </div>

        {/* CSS animated orb — replaces broken v0.dev video */}
        <div className="absolute top-1/2 right-[-15%] -translate-y-1/2 w-[600px] h-[600px] z-5 opacity-20 pointer-events-none">
          <div className="w-full h-full rounded-full border border-blue-500/20 animate-[spin_30s_linear_infinite]" />
          <div className="absolute inset-12 rounded-full border border-cyan-500/20 animate-[spin_20s_linear_infinite_reverse]" />
          <div className="absolute inset-32 rounded-full bg-blue-600/10 blur-3xl animate-pulse" />
        </div>
      </div>

      {/* Right Panel */}
      <div className="flex-1 flex flex-col items-center justify-center p-12 bg-[#f8fafc] relative">
        <div className="lg:hidden absolute top-10 left-10 flex items-center gap-3">
          <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center text-white font-black">TB</div>
          <h1 className="text-xl font-black text-slate-950 font-[Poppins]">Vision Pro</h1>
        </div>

        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="w-full max-w-[460px]">
          <div className="mb-12 text-center lg:text-left">
            <div className="flex items-center gap-3 mb-4 justify-center lg:justify-start">
              <div className="p-2 bg-blue-600/10 rounded-lg text-blue-600">{titles[mode].icon}</div>
              <p className="text-xs font-black text-blue-600 uppercase tracking-widest">{titles[mode].sub}</p>
            </div>
            <h3 className="text-4xl font-black text-slate-900 tracking-tight font-[Poppins] mb-4">{titles[mode].heading}</h3>
            <p className="text-slate-500 font-medium leading-relaxed">
              {mode === 'login' ? 'Please authorize your session to access clinical records.'
                : mode === 'signup' ? 'Create your professional medical profile to start screenings.'
                : 'Enter your registered email and we will send reset instructions.'}
            </p>
          </div>

          {resetSent && mode === 'reset' ? (
            <div className="text-center space-y-6 py-8">
              <div className="w-16 h-16 bg-emerald-100 rounded-full flex items-center justify-center mx-auto mb-2">
                <Mail className="text-emerald-600" size={28} />
              </div>
              <p className="font-black text-slate-800 text-xl">Reset Email Sent</p>
              
              <div className="p-4 bg-emerald-50 border border-emerald-100 rounded-xl text-left">
                 <p className="text-emerald-800 text-sm leading-relaxed">
                   Reset email sent to <strong>{formData.email}</strong>. Check your inbox and spam folder. The link expires in 30 minutes.
                 </p>
              </div>

              <div className="flex flex-col items-center gap-4 mt-8">
                <button 
                  onClick={handleSubmit} 
                  disabled={resendCountdown > 0 || loading}
                  className="btn-primary w-full max-w-[200px] py-3 text-sm disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {resendCountdown > 0 ? `Resend in ${resendCountdown}s` : 'Resend Email'}
                </button>
                <button onClick={() => { setMode('login'); setResetSent(false); setResendCountdown(0); }} className="text-slate-500 font-bold text-sm hover:text-blue-600 underline-offset-4 hover:underline transition-all">
                  Back to Login
                </button>
              </div>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Error Banner — specific messages */}
              {error && (
                <div className="p-4 bg-red-50 dark:bg-red-950/20 border border-red-300 dark:border-red-800 border-l-4 border-l-red-600 rounded-xl mb-4 flex items-start gap-3 shadow-md shadow-red-500/5">
                  <span className="text-lg shrink-0">❌</span>
                  <div className="flex-1">
                    <div className="text-red-700 dark:text-red-400 font-bold text-sm">
                      Access Authorization Failed
                    </div>
                    <div className="text-red-600 dark:text-red-400 text-xs font-semibold mt-1 leading-relaxed">
                      {error}
                    </div>
                    {(error.includes('starting up') || error.includes('sleep')) && (
                      <button
                        type="button"
                        onClick={handleAutoRetry}
                        className="mt-3 px-3 py-1.5 bg-red-600 hover:bg-red-700 active:scale-95 text-white text-[11px] font-black uppercase tracking-widest rounded-lg cursor-pointer transition-all shadow"
                      >
                        Auto-retry in 30s
                      </button>
                    )}
                  </div>
                </div>
              )}

              {/* Status banner */}
              {status && (
                <div className="p-3.5 bg-blue-50 dark:bg-blue-950/20 border border-blue-300 dark:border-blue-800 rounded-xl mb-4 text-blue-700 dark:text-blue-400 flex items-center gap-3 shadow-sm font-semibold text-xs animate-pulse">
                  <span className="inline-block w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin shrink-0" />
                  {status}
                </div>
              )}

              <AnimatePresence mode="wait">
                {mode === 'signup' && (
                  <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }} className="space-y-2 overflow-hidden pb-2">
                    <label className="text-[11px] font-black text-slate-400 uppercase tracking-widest px-1 block">Full Medical Name</label>
                    <input type="text" required className="input-field" placeholder="e.g. Dr. Sarah Jenkins"
                      value={formData.full_name} onChange={(e) => setFormData({ ...formData, full_name: e.target.value })} />
                  </motion.div>
                )}
              </AnimatePresence>

              <div className="space-y-2">
                <label className="text-[11px] font-black text-slate-400 uppercase tracking-widest px-1 block">Clinical Email</label>
                <input type="email" required className="input-field" placeholder="name@hospital.org"
                  value={formData.email} onChange={(e) => setFormData({ ...formData, email: e.target.value })} />
              </div>

              {(mode === 'login' || mode === 'signup') && (
                <div className="space-y-2">
                  <div className="flex justify-between items-center px-1">
                    <label className="text-[11px] font-black text-slate-400 uppercase tracking-widest">Secure Keyphrase</label>
                    {mode === 'login' && (
                      <button type="button" onClick={() => setMode('reset')} className="text-[11px] font-black text-blue-600 hover:text-blue-700 uppercase tracking-tighter">
                        Recovery?
                      </button>
                    )}
                  </div>
                  <input type="password" required className="input-field" placeholder="••••••••••••"
                    value={formData.password} onChange={(e) => setFormData({ ...formData, password: e.target.value })} />
                </div>
              )}

              <button type="submit" disabled={loading}
                className="btn-primary w-full mt-4 py-4.5 flex items-center justify-center gap-3 active:scale-[0.98] disabled:opacity-70 disabled:cursor-not-allowed group">
                {loading ? (
                    <div className="flex flex-col items-center gap-1">
                        <Activity className="animate-spin" size={20} />
                        <span className="text-[10px] font-bold uppercase tracking-widest animate-pulse">
                          Initializing Server... {retryStatus}
                        </span>
                    </div>
                ) : (
                  <>
                    <span className="font-black uppercase tracking-widest text-[13px]">
                      {mode === 'login' ? 'Authorize Access' : mode === 'signup' ? 'Initialize Account' : 'Send Reset Email'}
                    </span>
                    <ArrowRight size={20} className="group-hover:translate-x-1 transition-transform" />
                  </>
                )}
              </button>
            </form>
          )}

          <div className="mt-10 flex flex-col items-center gap-6">
            <div className="w-full h-px bg-slate-200" />
            <p className="text-[13px] font-bold text-slate-500">
              {mode === 'login' ? 'Need a professional ID?' : mode === 'signup' ? 'Already registered?' : 'Back to authentication?'}
              <button onClick={() => { setMode(mode === 'login' ? 'signup' : 'login'); setResetSent(false); setResendCountdown(0); }}
                className="ml-2 text-blue-600 hover:underline font-black">
                {mode === 'login' ? 'Request Profile' : 'Access System'}
              </button>
            </p>
          </div>

          <div className="mt-12 flex items-center justify-center gap-10 opacity-30 hover:opacity-100 transition-all cursor-default">
            <ShieldCheck size={26} /><Lock size={26} /><HeartPulse size={26} /><Zap size={26} />
          </div>
        </motion.div>
      </div>
    </div>
  );
}

