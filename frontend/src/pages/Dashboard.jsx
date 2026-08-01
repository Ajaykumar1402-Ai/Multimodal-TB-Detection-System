import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { getDashboardStats } from '../services/api';
import { toast } from 'react-hot-toast';
import { motion } from 'framer-motion';
import {
  ChevronRight, Plus, Zap,
  BrainCircuit, ShieldCheck, TrendingUp,
  ActivitySquare, LayoutGrid, PieChart,
  Microscope, Activity
} from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const itemVariants = {
  hidden:  { y: 20, opacity: 0 },
  visible: { y: 0, opacity: 1, transition: { duration: 0.5, ease: 'easeOut' } },
};

const ClinicalPulseVisual = () => (
  <div className="w-full h-full relative overflow-hidden bg-slate-950 flex items-center justify-center">
    <div className="absolute inset-0 opacity-20">
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-600 blur-[120px] rounded-full animate-pulse" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-cyan-500 blur-[120px] rounded-full animate-pulse" style={{ animationDelay: '1s' }} />
    </div>
    <div className="relative z-10 w-full max-w-lg aspect-square">
      <div className="absolute inset-0 border-[1px] border-blue-500/20 rounded-full animate-[spin_20s_linear_infinite]" />
      <div className="absolute inset-8 border-[1px] border-cyan-500/10 rounded-full animate-[spin_15s_linear_infinite_reverse]" />
      <div className="absolute inset-0 flex items-center justify-center">
        <motion.div
          animate={{ scale: [1, 1.05, 1], rotate: [0, 5, -5, 0] }}
          transition={{ duration: 6, repeat: Infinity, ease: 'easeInOut' }}
          className="w-48 h-48 bg-blue-600/10 rounded-[3rem] border border-blue-400/30 flex items-center justify-center shadow-[0_0_50px_rgba(37,99,235,0.2)]"
        >
          <BrainCircuit size={80} className="text-blue-400 opacity-60" />
        </motion.div>
      </div>
      {[0, 60, 120, 180, 240, 300].map((deg, i) => (
        <motion.div
          key={i}
          className="absolute w-3 h-3 bg-blue-400 rounded-full shadow-[0_0_15px_#3b82f6]"
          animate={{ opacity: [0.3, 1, 0.3], scale: [0.8, 1.2, 0.8] }}
          transition={{ duration: 3, delay: i * 0.5, repeat: Infinity }}
          style={{ top: '50%', left: '50%', transform: `rotate(${deg}deg) translate(200px) rotate(-${deg}deg)` }}
        />
      ))}
    </div>
  </div>
);

export default function Dashboard() {
  const navigate = useNavigate();
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const data = await getDashboardStats();
        setStats(data);
      } catch (err) {
        toast.error('Clinical synchronization failed');
      } finally {
        setLoading(false);
      }
    };
    fetchStats();
  }, []);

  if (loading || !stats) {
    return (
      <div className="h-full flex items-center justify-center min-h-[400px]">
        <div className="w-16 h-16 border-4 border-blue-600 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  const total = stats.total || 1;
  const highRiskPct = Math.round(((stats.highRisk || 0) / total) * 100);
  const resolvedPct = Math.round(((stats.resolved || 0) / total) * 100);
  const mediumPct   = Math.max(0, 100 - highRiskPct - resolvedPct);

  const riskAllocation = [
    { l: 'High Risk Detected', p: highRiskPct, c: 'bg-rose-500',    tc: 'text-rose-500'    },
    { l: 'Under Observation',  p: mediumPct,   c: 'bg-amber-500',   tc: 'text-amber-500'   },
    { l: 'Cleared / Negative', p: resolvedPct, c: 'bg-emerald-500', tc: 'text-emerald-500' },
  ];

  return (
    <div className="p-4 md:p-8 lg:p-10 max-w-[1700px] mx-auto font-sans text-slate-900 bg-[#f8fafc] min-h-screen">

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
        {[
          { label: 'Total Screenings', val: stats.total    || 0,  icon: Microscope,   color: 'text-blue-600',    bg: 'bg-blue-50'    },
          { label: 'High Risk Cases',  val: stats.highRisk || 0,  icon: ShieldCheck,  color: 'text-rose-600',    bg: 'bg-rose-50'    },
          { label: 'Cases Cleared',    val: stats.resolved || 0,  icon: Activity,     color: 'text-emerald-600', bg: 'bg-emerald-50' },
          { label: 'Model Version',    val: 'v2.5.0',             icon: BrainCircuit, color: 'text-indigo-600',  bg: 'bg-indigo-50'  },
        ].map((s, i) => (
          <motion.div key={i} variants={itemVariants} initial="hidden" animate="visible"
            className="bg-white p-7 rounded-[2rem] border border-slate-200/60 shadow-sm flex items-center gap-6 group hover:shadow-2xl hover:border-blue-200 transition-all cursor-default">
            <div className={`w-16 h-16 ${s.bg} ${s.color} rounded-2xl flex items-center justify-center transition-transform group-hover:scale-110 shadow-inner`}>
              <s.icon size={28} />
            </div>
            <div>
              <p className="text-[10px] font-black text-slate-400 uppercase tracking-[0.15em] mb-1.5">{s.label}</p>
              <p className="text-3xl font-black text-slate-950 tracking-tighter">{s.val}</p>
            </div>
          </motion.div>
        ))}
      </div>

      <div className="flex flex-col xl:grid xl:grid-cols-12 gap-10">

        <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} className="xl:col-span-8 space-y-10">

          <div className="relative rounded-[3rem] md:rounded-[3.5rem] bg-gradient-to-br from-[#0f172a] via-[#1e293b] to-black overflow-hidden h-[450px] md:h-[650px] shadow-[0_40px_100px_rgba(0,0,0,0.15)] border border-white/5 group">
            <div className="absolute top-8 left-8 md:top-12 md:left-12 z-10 max-w-md">
              <div className="px-4 py-2 bg-blue-500/10 border border-blue-400/20 rounded-full inline-flex items-center gap-2 mb-4 md:mb-6 backdrop-blur-2xl">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse shadow-[0_0_10px_#3b82f6]" />
                <p className="text-[10px] font-black text-blue-400 uppercase tracking-widest leading-none">Clinical Pulse Active</p>
              </div>
              <h2 className="text-4xl md:text-6xl font-black text-white leading-[1.05] font-[Poppins] tracking-tight mb-4">
                Clinical <br /><span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400">Environment</span>
              </h2>
              <p className="text-slate-400 font-medium text-[11px] md:text-sm leading-relaxed max-w-[200px] md:max-w-xs">
                Real-time neural synthesis of multimodal biometric data streams.
              </p>
            </div>

            <div className="absolute inset-0 z-0 group-hover:scale-105 transition-transform duration-[4s] ease-out">
              <ClinicalPulseVisual />
            </div>

            <div className="absolute top-8 right-8 md:top-12 md:right-12 z-10">
              <button onClick={() => navigate('/app/diagnosis')}
                className="bg-white/5 hover:bg-white/15 backdrop-blur-2xl border border-white/10 p-4 md:p-5 rounded-2xl text-white transition-all shadow-2xl group/btn" title="New Analysis">
                <Plus size={24} className="group-hover/btn:scale-110 transition-transform" />
              </button>
            </div>

            <div className="absolute bottom-8 left-8 right-8 md:bottom-12 md:left-12 md:right-12 z-10 grid grid-cols-1 md:grid-cols-3 gap-3 md:gap-6">
              {[
                { l: 'Radiology Stream', v: '99.9% Integrity',  i: ActivitySquare, c: 'text-blue-400'  },
                { l: 'AI Orchestration', v: 'Active Consensus',  i: ShieldCheck,    c: 'text-cyan-400'  },
                { l: 'System Latency',   v: '4ms Global',        i: Zap,            c: 'text-amber-400' },
              ].map((m, i) => (
                <div key={i} className="bg-white/5 backdrop-blur-3xl border border-white/10 p-4 md:p-6 rounded-[1.5rem] md:rounded-[2.5rem] flex items-center gap-4 md:gap-5 hover:bg-white/10 transition-all">
                  <div className={`p-3 md:p-4 bg-white/5 rounded-xl md:rounded-2xl ${m.c} shadow-xl`}><m.i size={20} /></div>
                  <div>
                    <p className="text-[8px] md:text-[10px] font-black text-slate-500 uppercase tracking-widest mb-0.5 md:mb-1">{m.l}</p>
                    <p className="text-[11px] md:text-sm font-black text-white">{m.v}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
            <div className="bg-white p-10 rounded-[3.5rem] border border-slate-200/60 shadow-xl overflow-hidden">
              <div className="flex justify-between items-center mb-10">
                <div className="flex items-center gap-4">
                  <div className="p-3 bg-blue-50 text-blue-600 rounded-2xl"><TrendingUp size={20} /></div>
                  <h3 className="text-xs font-black text-slate-800 uppercase tracking-widest">Monthly Screenings</h3>
                </div>
              </div>
              <div className="h-48">
                <ResponsiveContainer width="100%" height={192}>
                  <BarChart data={stats.chartData || []} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                    <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 10, fontWeight: 900, fill: '#94a3b8' }} dy={10} />
                    <YAxis hide />
                    <Tooltip cursor={{ fill: '#f8fafc' }} contentStyle={{ borderRadius: '24px', border: 'none', boxShadow: '0 20px 50px rgba(0,0,0,0.08)', padding: '20px' }} />
                    <Bar dataKey="screenings" radius={[8, 8, 0, 0]} barSize={24}>
                      {(stats.chartData || []).map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={index === (stats.chartData?.length - 1) ? '#2563eb' : '#e2e8f0'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-white p-10 rounded-[3.5rem] border border-slate-200/60 shadow-xl flex flex-col justify-between">
              <div>
                <div className="flex items-center gap-4 mb-8">
                  <div className="p-3 bg-rose-50 text-rose-600 rounded-2xl"><PieChart size={20} /></div>
                  <h3 className="text-xs font-black text-slate-800 uppercase tracking-widest">Risk Allocation</h3>
                </div>
              </div>
              <div className="space-y-6">
                {riskAllocation.map((item, idx) => (
                  <div key={idx} className="space-y-2">
                    <div className="flex justify-between items-center text-[11px] font-black uppercase tracking-wider">
                      <span className="text-slate-400">{item.l}</span>
                      <span className={item.tc}>{item.p}%</span>
                    </div>
                    <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
                      <motion.div 
                        initial={{ width: 0 }} 
                        animate={{ width: `${item.p}%` }} 
                        className={`h-full ${item.c} rounded-full`} 
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </motion.div>

        <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="xl:col-span-4 space-y-10">
          <div className="bg-white rounded-[3.5rem] border border-slate-200/60 shadow-xl overflow-hidden flex flex-col h-[450px] md:h-[650px]">
            <div className="p-10 border-b border-slate-100 flex justify-between items-center bg-slate-50/50">
              <div className="flex items-center gap-4">
                <div className="p-3 bg-indigo-50 text-indigo-600 rounded-2xl"><LayoutGrid size={20} /></div>
                <h3 className="text-xs font-black text-slate-800 uppercase tracking-widest">Global Registry</h3>
              </div>
              <button 
                onClick={() => navigate('/app/history')}
                className="text-[10px] font-black text-blue-600 uppercase tracking-widest hover:translate-x-1 transition-transform flex items-center gap-2"
              >
                Full Access <ChevronRight size={14} />
              </button>
            </div>
            
            <div className="flex-1 overflow-y-auto p-6 space-y-4 custom-scrollbar">
              {(stats.recent || []).map((r, i) => (
                <div key={i} className="p-5 bg-white border border-slate-100 rounded-3xl flex items-center gap-4 hover:border-blue-100 hover:shadow-lg transition-all group cursor-default">
                  <div className="w-12 h-12 rounded-2xl bg-slate-50 flex items-center justify-center text-slate-400 group-hover:bg-blue-600 group-hover:text-white transition-all font-black">
                    {r.patient_name ? r.patient_name[0].toUpperCase() : 'P'}
                  </div>
                  <div className="flex-1">
                    <p className="text-sm font-black text-slate-900 line-clamp-1">{r.patient_name}</p>
                    <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mt-1">{r.risk_level} Priority</p>
                  </div>
                  <div className={`w-2.5 h-2.5 rounded-full ${r.risk_level === 'High' ? 'bg-rose-500' : r.risk_level === 'Medium' ? 'bg-amber-500' : 'bg-emerald-500'}`} />
                </div>
              ))}
              {(!stats.recent || stats.recent.length === 0) && (
                 <div className="h-full flex flex-col items-center justify-center grayscale opacity-20 py-10">
                    <Activity size={40} />
                    <p className="text-[10px] font-black uppercase mt-4">No Recent Records</p>
                 </div>
              )}
            </div>
          </div>
        </motion.div>

      </div>
    </div>
  );
}
