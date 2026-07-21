import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { getAllDiagnoses } from '../services/api';
import {
  Search, FileText, Download, Filter,
  MessageSquare, RefreshCcw, Calendar,
  BrainCircuit
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { toast } from 'react-hot-toast';

const containerVariants = {
  hidden:  { opacity: 0 },
  visible: { opacity: 1, transition: { staggerChildren: 0.05 } },
};
const itemVariants = {
  hidden:  { y: 10, opacity: 0 },
  visible: { y: 0, opacity: 1, transition: { duration: 0.4, ease: 'easeOut' } },
};

export default function History() {
  const location = useLocation();
  const navigate = useNavigate();
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filterRisk, setFilterRisk] = useState('all');

  const queryParams = new URLSearchParams(location.search);
  const initialQuery = queryParams.get('query') || '';
  const [search, setSearch] = useState(initialQuery);

  useEffect(() => {
    const delayDebounceFn = setTimeout(() => {
      fetchHistory(search);
    }, 500);
    return () => clearTimeout(delayDebounceFn);
  }, [search]);

  const fetchHistory = async (query = '') => {
    setLoading(true);
    try {
      const data = await getAllDiagnoses(query);
      setRecords(data);
    } catch (err) {
      toast.error('Clinical records unreachable');
    } finally {
      setLoading(false);
    }
  };

  const handleExport = () => {
    if (records.length === 0) {
      toast.error('No records to export');
      return;
    }
    const headers = ['ID', 'Patient Name', 'Date', 'Risk Level', 'Confidence (%)'];
    const rows = records.map((r) => [
      r.id,
      `"${r.patient_name}"`,
      r.date,
      r.risk_level,
      (parseFloat(r.final_prob || 0) * 100).toFixed(0),
    ]);
    const csv = [headers, ...rows].map((row) => row.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `tb_vision_records_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success(`Exported ${records.length} records`);
  };

  const getRiskBadge = (risk) => {
    const styles = {
      High:     'bg-rose-100 text-rose-700 border-rose-200',
      Medium:   'bg-amber-100 text-amber-700 border-amber-200',
      Low:      'bg-emerald-100 text-emerald-700 border-emerald-200',
      REDACTED: 'bg-slate-900 text-slate-100 border-slate-900',
    };
    return (
      <span className={`px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-widest border ${styles[risk] || styles.Low}`}>
        {risk || 'Negative'}
      </span>
    );
  };

  const filteredRecords = filterRisk === 'all'
    ? records
    : records.filter((r) => r.risk_level === filterRisk);

  return (
    <div className="p-4 md:p-8 lg:p-12 max-w-[1600px] mx-auto font-sans text-slate-900 bg-[#f8fafc] min-h-screen">

      <header className="mb-10 flex flex-col md:flex-row md:items-end justify-between gap-6">
        <div>
          <div className="flex items-center gap-3 mb-3">
            <BrainCircuit className="text-blue-600" size={24} />
            <h1 className="text-3xl font-black text-slate-900 tracking-tight font-[Poppins]">Clinical Archives</h1>
          </div>
          <p className="text-slate-500 font-medium text-[15px]">Central registry for multimodal Tuberculosis diagnostic records.</p>
        </div>
        <div className="flex gap-3">
          <button onClick={() => fetchHistory(search)}
            className="p-3 bg-white border border-slate-200 rounded-xl text-slate-400 hover:text-blue-600 hover:border-blue-100 transition-all shadow-sm group" title="Refresh">
            <RefreshCcw size={20} className="group-active:rotate-180 transition-transform duration-500" />
          </button>
          <button onClick={handleExport}
            className="flex items-center gap-2 px-5 py-3 bg-blue-600 text-white rounded-xl font-black text-xs uppercase tracking-widest hover:bg-blue-700 transition-all shadow-lg shadow-blue-600/20 active:scale-95">
            <Download size={16} /> Export CSV
          </button>
        </div>
      </header>

      <div className="glass-panel p-4 mb-8 flex flex-col lg:flex-row gap-4">
        <div className="flex-1 relative group">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 group-focus-within:text-blue-500 transition-colors" size={18} />
          <input type="text" placeholder="Search by patient name..."
            value={search} onChange={(e) => setSearch(e.target.value)}
            className="w-full bg-slate-50 border border-slate-200 rounded-xl py-3.5 pl-12 pr-4 text-slate-900 placeholder:text-slate-400 focus:border-blue-500 focus:ring-4 focus:ring-blue-500/5 outline-none transition-all font-medium" />
        </div>
        <div className="flex gap-2">
          {[
            { v: 'all',    l: 'All'      },
            { v: 'High',   l: 'High Risk' },
            { v: 'Medium', l: 'Medium'   },
            { v: 'Low',    l: 'Cleared'  },
          ].map((f) => (
            <button key={f.v} onClick={() => setFilterRisk(f.v)}
              className={`flex-1 lg:flex-none px-4 py-3 rounded-xl text-[11px] font-black uppercase tracking-widest transition-all border ${
                filterRisk === f.v
                  ? 'bg-blue-600 text-white border-blue-600 shadow-lg shadow-blue-600/20'
                  : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'}`}>
              {f.l}
            </button>
          ))}
        </div>
      </div>

      <motion.div variants={containerVariants} initial="hidden" animate="visible"
        className="glass-panel overflow-hidden border-slate-200 shadow-xl">
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="bg-slate-50/80 text-slate-400 text-[10px] uppercase tracking-[0.2em] border-b border-slate-200">
                <th className="px-8 py-5 font-black">Clinical Profile</th>
                <th className="px-8 py-5 font-black">Registry Date</th>
                <th className="px-8 py-5 font-black">ML Assessment</th>
                <th className="px-8 py-5 font-black">Confidence</th>
                <th className="px-8 py-5 font-black text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 bg-white">
              {filteredRecords.map((r) => (
                <motion.tr key={r.id} variants={itemVariants}
                  className="hover:bg-blue-50/30 transition-all group cursor-pointer"
                  onClick={() => navigate(`/app/history?query=${encodeURIComponent(r.patient_name)}`)}>
                  <td className="px-8 py-6">
                    <div className="flex items-center gap-4">
                      <div className="w-12 h-12 rounded-2xl bg-slate-100 flex items-center justify-center text-slate-400 group-hover:bg-blue-600 group-hover:text-white transition-all duration-500 font-black text-lg">
                        {r.patient_name ? r.patient_name[0].toUpperCase() : 'P'}
                      </div>
                      <div>
                        <p className="font-black text-slate-900 group-hover:text-blue-600 transition-colors font-[Poppins] tracking-tight">{r.patient_name}</p>
                        <p className="text-[10px] text-slate-400 font-black uppercase tracking-widest mt-0.5">Ref: #ID-{r.id?.toString().padStart(4, '0')}</p>
                      </div>
                    </div>
                  </td>
                  <td className="px-8 py-6">
                    <div className="flex items-center gap-2 text-slate-500 font-semibold text-sm">
                      <Calendar size={14} className="opacity-40" />
                      {new Date(r.created_at || r.date).toLocaleDateString('en-US', { day: 'numeric', month: 'short', year: 'numeric' })}
                    </div>
                  </td>
                  <td className="px-8 py-6">{getRiskBadge(r.risk_level)}</td>
                  <td className="px-8 py-6">
                    <div className="flex items-center gap-3">
                      <div className="w-24 h-2 bg-slate-100 rounded-full overflow-hidden hidden sm:block">
                        <motion.div initial={{ width: 0 }} animate={{ width: `${(r.final_prob || 0) * 100}%` }}
                          className={`h-full rounded-full ${r.final_prob > 0.7 ? 'bg-rose-500' : 'bg-blue-500'}`} />
                      </div>
                      <span className="font-black text-slate-900 text-sm">
                        {(parseFloat(r.final_prob || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                  </td>
                  <td className="px-8 py-6 text-right" onClick={(e) => e.stopPropagation()}>
                    <div className="flex justify-end gap-2">
                      {r.pdf_url ? (
                        <a href={r.pdf_url} target="_blank" rel="noopener noreferrer"
                          className="p-3 bg-slate-50 rounded-xl text-slate-400 hover:bg-blue-600 hover:text-white transition-all shadow-sm active:scale-95" title="Download Report">
                          <Download size={18} />
                        </a>
                      ) : (
                        <button className="p-3 bg-slate-50 rounded-xl text-slate-200 cursor-not-allowed" title="No Report Yet">
                          <Download size={18} />
                        </button>
                      )}
                      <button
                        className="p-3 bg-slate-50 rounded-xl text-slate-400 hover:bg-blue-600 hover:text-white transition-all shadow-sm active:scale-95"
                        onClick={() => toast.success(r.recommendations || 'No recommendations noted.')}
                        title="View Recommendations">
                        <MessageSquare size={18} />
                      </button>
                    </div>
                  </td>
                </motion.tr>
              ))}
              {!loading && filteredRecords.length === 0 && (
                <tr>
                  <td colSpan={5} className="p-24 text-center">
                    <div className="flex flex-col items-center grayscale opacity-30">
                      <FileText size={48} className="mb-4" />
                      <p className="font-black text-slate-400 uppercase tracking-widest text-xs">No Clinical Records Found</p>
                    </div>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </motion.div>

      <AnimatePresence>
        {loading && records.length === 0 && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="h-96 flex flex-col items-center justify-center space-y-4">
            <div className="w-12 h-12 border-4 border-slate-200 border-t-blue-600 rounded-full animate-spin" />
            <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Querying Registry...</p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

  
          
     
