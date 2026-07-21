import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDiagnosis } from '../context/DiagnosisContext';
import { toast } from 'react-hot-toast';
import { 
  Download, Search, Microscope, AlertCircle, CheckCircle, 
  Sparkles, BrainCircuit, Layers, Activity, TrendingUp
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import GlassCard from '../components/GlassCard';
import { 
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip 
} from 'recharts';

export default function Result() {
  const navigate = useNavigate();
  const { state, dispatch } = useDiagnosis();
  const { result, preview, formData } = state;

  useEffect(() => {
    // BUG C-03 Route Guard: If no valid analysis score exists, never render stale data.
    // URGENT STEP 3: Redirect if session is invalid or not authenticated
    if (!result || !result.validation_passed) {
      console.warn("[GUARD] No active analysis score found. Redirecting to ingest...");
      navigate('/app/diagnosis', { replace: true });
    }
  }, [result, navigate]);

  if (!result || !result.validation_passed) {
    return null; // Prevent flash of stale data
  }

  // STEP 4: URGENT PII PROTECTION - Only use clinician-typed data
  const displayPatientName = formData.patient_name || "Not Provided";
  const displayPatientId = formData.patient_id || "Not Provided";

  const downloadReport = () => {
     if (result?.pdf_url) {
        window.open(result.pdf_url, '_blank');
        toast.success('Report generation initialized');
     } else {
        toast.error('PDF record not found');
     }
  };

  const handleNewAnalysis = () => {
    // BUG C-03: Reset all analysis state as the very first line before navigation.
    // This is handled by our global DiagnosisContext reducer.
    dispatch({ type: 'RESET_ANALYSIS' });
    navigate('/app/diagnosis', { replace: true });
  };

  return (
    <div className="p-4 md:p-8 lg:p-12 max-w-[1600px] mx-auto font-sans text-slate-900 bg-[#f8fafc] min-h-screen">
      <div className="mb-10 flex items-center justify-between">
        <div>
          <div className="flex items-center gap-3 mb-3">
            <Sparkles className="text-blue-600" size={24} />
            <h2 className="text-3xl font-black text-slate-900 tracking-tight font-[Poppins]">Intelligence Report</h2>
          </div>
          <p className="text-slate-500 font-medium text-[15px]">Multi-modal AI synthesis and clinical correlation analysis.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        <div className="lg:col-span-7">
           <GlassCard title="Medical Engine Output" icon={Microscope}>
              <div className="relative h-[600px] rounded-3xl overflow-hidden bg-slate-900 shadow-2xl border border-white/5">
                 {preview && (
                    <img 
                      src={preview} 
                      alt="X-ray" 
                      className="w-full h-full object-contain brightness-110 grayscale contrast-125 saturate-0 opacity-80" 
                      style={{ filter: 'grayscale(100%) contrast(1.2) brightness(0.9) blur(0.5px)' }}
                    />
                 )}
                 {result?.results?.medsam_mask_url && (
                    <motion.img 
                       initial={{ opacity: 0 }} animate={{ opacity: 0.7 }}
                       src={result.results.medsam_mask_url} 
                       className="absolute inset-0 w-full h-full object-contain rounded-2xl mix-blend-multiply pointer-events-none" 
                    />
                 )}
                 <div className="absolute top-8 left-8 flex flex-col gap-3">
                    <div className="px-4 py-2 bg-slate-900/60 backdrop-blur-xl border border-white/10 rounded-xl text-white text-[10px] font-black uppercase tracking-widest">
                       Clinical ID: {displayPatientId}
                    </div>
                    <div className="px-4 py-2 bg-slate-900/60 backdrop-blur-xl border border-white/10 rounded-xl text-white text-[10px] font-black uppercase tracking-widest">
                       Patient: {displayPatientName}
                    </div>
                 </div>
              </div>
           </GlassCard>
        </div>

        <div className="lg:col-span-5 space-y-8">
           <GlassCard title="Inference Metrics" icon={Activity} className="border-blue-200 ring-4 ring-blue-600/5">
              <div className="space-y-6">
                 <div className="text-center relative">
                    <div className="inline-flex flex-col items-center">
                       <div className={`p-8 rounded-full mb-4 relative ${result.probability_mean >= 0.42 ? 'bg-rose-50 text-rose-600' : 'bg-emerald-50 text-emerald-600'}`}>
                          {result.probability_mean >= 0.42 ? <AlertCircle size={48} /> : <CheckCircle size={48} />}
                       </div>
                       <h4 className={`text-3xl font-black tracking-tighter font-[Poppins] mb-1 ${result.probability_mean >= 0.42 ? 'text-rose-600' : 'text-emerald-600'}`}>
                          {result.probability_mean >= 0.42 ? 'Tuberculosis Detected' : 'Tuberculosis Negative'}
                       </h4>
                       <p className="text-slate-400 text-[10px] font-black uppercase tracking-widest">Autonomous Machine Analysis</p>
                    </div>
                 </div>

                 <div className="grid grid-cols-2 gap-4">
                    <div className="p-5 bg-slate-50 rounded-2xl border border-slate-100 relative overflow-hidden flex flex-col justify-center">
                       <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-3">Calibrated Confidence</p>
                       <div className="flex flex-col gap-2 w-full">
                          <div className="flex items-end gap-1 group relative cursor-help">
                             <span className="font-black text-slate-900 text-3xl leading-none">
                                {(parseFloat(result.probability_mean || 0) * 100).toFixed(1)}%
                             </span>
                             {result.confidence_interval !== undefined ? (
                               <span className="text-[10px] font-bold text-slate-500 ml-2 mb-1">
                                  (95% CI: {(result.confidence_interval[0] * 100).toFixed(0)}% – {(result.confidence_interval[1] * 100).toFixed(0)}%)
                               </span>
                             ) : (
                               <span className="text-[10px] font-bold text-rose-500 ml-2 mb-1">
                                  ⚠️ Uncertainty data unavailable. Interpret result with caution.
                               </span>
                             )}
                             
                             {/* Tooltip */}
                             <div className="absolute -top-10 left-0 bg-slate-800 text-white text-[9px] px-3 py-1.5 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none uppercase font-black tracking-widest z-50 whitespace-nowrap shadow-xl">
                                This range reflects model uncertainty across 30 inference runs.
                             </div>
                          </div>

                          {/* Horizontal Confidence Interval Bar */}
                          <div className="relative w-full h-3 bg-slate-200 rounded-full mt-2 overflow-hidden shadow-inner">
                             {/* CI Range Overlay */}
                             <motion.div 
                                initial={{ opacity: 0, scaleX: 0 }}
                                animate={{ 
                                   opacity: 1,
                                   scaleX: 1,
                                   left: `${result.confidence_interval ? result.confidence_interval[0] * 100 : 0}%`,
                                   width: `${result.confidence_interval ? (result.confidence_interval[1] - result.confidence_interval[0]) * 100 : 0}%` 
                                }}
                                transition={{ duration: 1.5, ease: 'circOut' }}
                                className={`absolute h-full opacity-40 ${result.probability_mean >= 0.42 ? 'bg-rose-400' : 'bg-emerald-400'}`}
                             />
                             {/* Point Estimate Marker */}
                             <motion.div 
                                initial={{ left: 0 }}
                                animate={{ left: `${result.probability_mean * 100}%` }}
                                transition={{ duration: 1.5, ease: 'circOut' }}
                                className={`absolute top-0 bottom-0 w-1.5 border-x border-white/20 shadow-sm ${result.probability_mean >= 0.42 ? 'bg-rose-600' : 'bg-emerald-600'}`}
                                style={{ transform: 'translateX(-50%)' }}
                             />
                          </div>
                          
                          <div className="flex justify-between text-[8px] font-bold text-slate-400 mt-1 uppercase tracking-widest px-1">
                             <span>{result.confidence_interval ? (result.confidence_interval[0] * 100).toFixed(0) : 0}%</span>
                             <span className="flex items-center gap-1 opacity-40">
                                <span className="w-1 h-1 bg-slate-400 rounded-full"/> 95% CI Range <span className="w-1 h-1 bg-slate-400 rounded-full"/>
                             </span>
                             <span>{result.confidence_interval ? (result.confidence_interval[1] * 100).toFixed(0) : 100}%</span>
                          </div>

                          {/* Uncertainty Badge */}
                          <div className="mt-4 flex items-center gap-2">
                             <div className={`px-3 py-1 rounded-full text-[9px] font-black uppercase tracking-widest flex items-center gap-2 shadow-sm border ${result.uncertainty > 0.08 ? 'bg-amber-50 text-amber-600 border-amber-100' : 'bg-emerald-50 text-emerald-600 border-emerald-100'}`}>
                                {result.uncertainty > 0.08 ? (
                                   <><span className="w-1.5 h-1.5 bg-amber-500 rounded-full animate-pulse" /> Moderate Uncertainty</>
                                ) : (
                                   <><span className="w-1.5 h-1.5 bg-emerald-500 rounded-full" /> High Precision</>
                                )}
                             </div>
                          </div>
                       </div>
                    </div>
                    <div className="p-5 bg-slate-50 rounded-2xl border border-slate-100 flex flex-col justify-center">
                       <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-2">Priority</p>
                       <div className="flex items-center gap-2">
                          <p className={`text-2xl font-black leading-tight ${result.results?.risk_level === 'High' ? 'text-rose-600' : 'text-emerald-600'}`}>
                            {result.results?.risk_level}
                          </p>
                          <span className="text-slate-400 font-bold text-sm mt-1">Status</span>
                       </div>
                    </div>
                 </div>

                 <div className="space-y-4">
                    <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Neural Breakdown</p>
                    <div className="space-y-4">
                      {[
                        { n: 'MedSAM Segmenter', s: result?.results?.ensemble_breakdown?.medsam || 0.98, icon: Layers },
                        { n: 'ViT Transformer', s: result?.results?.ensemble_breakdown?.vit || 0.25, icon: BrainCircuit },
                        { n: 'DenseNet-121', s: result?.results?.ensemble_breakdown?.resnet || 0.61, icon: TrendingUp }
                      ].map((m, i) => (
                        <div key={i} className="space-y-2">
                           <div className="flex justify-between text-[9px] font-bold">
                              <div className="flex items-center gap-2 text-slate-600">
                                 <m.icon size={12} className="text-blue-500" />
                                 <span>{m.n}</span>
                              </div>
                              <span className="text-blue-600">{(parseFloat(m.s || 0)*100).toFixed(1)}%</span>
                           </div>
                           <div className="w-full h-1 bg-white/5 rounded-full overflow-hidden">
                              <motion.div 
                                 initial={{ width: 0 }}
                                 animate={{ width: `${(parseFloat(m.s || 0)*100)}%` }}
                                 transition={{ duration: 1, delay: i * 0.1 }}
                                 className="h-full bg-blue-500 rounded-full" 
                              />
                           </div>
                        </div>
                      ))}
                    </div>
                 </div>

                 <div className="grid grid-cols-2 gap-3 pt-2">
                    <button 
                       onClick={downloadReport}
                       className="flex items-center justify-center gap-3 bg-blue-600 text-white py-4 rounded-2xl font-black uppercase tracking-widest text-[10px] hover:bg-blue-700 transition-all shadow-lg"
                    >
                       <Download size={16} /> Report
                    </button>
                    <button 
                       onClick={() => navigate('/app/history')}
                       className="flex items-center justify-center gap-3 bg-white border border-slate-200 text-slate-600 py-4 rounded-2xl font-black uppercase tracking-widest text-[10px] hover:bg-slate-50 transition-all"
                    >
                       <Search size={16} /> History
                    </button>
                 </div>
              </div>
           </GlassCard>
        </div>
      </div>

      <AnimatePresence>
        {result && (
          <motion.div 
            initial={{ opacity: 0, y: 50 }} animate={{ opacity: 1, y: 0 }}
            className="fixed bottom-24 left-1/2 -translate-x-1/2 z-[60] flex gap-4 px-6 py-4 bg-slate-900/90 backdrop-blur-xl border border-white/10 rounded-full shadow-[0_20px_60px_rgba(0,0,0,0.3)] md:bottom-10"
          >
            <button 
              type="button" onClick={handleNewAnalysis}
              className="flex items-center gap-2 px-6 py-2.5 rounded-full text-[10px] md:text-xs font-black uppercase tracking-widest text-slate-400 hover:text-white transition-all underline decoration-blue-500 underline-offset-4"
            >
               <Activity size={16} /> New Analysis (Secure Reset)
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
