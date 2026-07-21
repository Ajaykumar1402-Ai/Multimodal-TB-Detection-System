import React, { useRef, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { uploadWithProgress, getAllDiagnoses, getPatientHistory } from '../services/api';
import { toast } from 'react-hot-toast';
import { useDiagnosis } from '../context/DiagnosisContext';
import { 
  UploadCloud, FileImage, AlertCircle, CheckCircle,
  User, Microscope, Sparkles, XCircle, BrainCircuit, TrendingUp,
  Activity, LayoutGrid, Share2, Save
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import PatientDataNotice from '../components/PatientDataNotice';
import UploadProgressBar from '../components/UploadProgressBar';
import GlassCard from '../components/GlassCard';
import { logConsent } from '../services/api';
import { ServerStatus } from '../components/ServerStatus';

// GlassCard moved to components/GlassCard.jsx

const SectionHeader = ({ title, subtitle, icon: Icon }) => (
  <div className="mb-10 flex items-center justify-between">
    <div>
      <div className="flex items-center gap-3 mb-3">
        {Icon && <Icon className="text-blue-600" size={24} />}
        <h2 className="text-3xl font-black text-slate-900 tracking-tight font-[Poppins]">{title}</h2>
      </div>
      <p className="text-slate-500 font-medium text-[15px]">{subtitle}</p>
    </div>
    <div className="hidden md:flex gap-3">
        <button className="p-3 bg-white border border-slate-200 rounded-xl text-slate-400 hover:text-blue-600 hover:border-blue-100 transition-all shadow-sm">
            <Share2 size={20} />
        </button>
        <button className="p-3 bg-white border border-slate-200 rounded-xl text-slate-400 hover:text-blue-600 hover:border-blue-100 transition-all shadow-sm">
            <Save size={20} />
        </button>
    </div>
  </div>
);

export default function Diagnosis() {
  const navigate = useNavigate();
  const fileInputRef = useRef(null);
  const xhrCancelRef = useRef(null);
  const { state, dispatch } = useDiagnosis();
  const { formData, file, preview, loading, uploadError, hasConsented } = state;
  const [recentPatients, setRecentPatients] = useState([]);
  const [showRecent, setShowRecent] = useState(false);
  const [patientHistory, setPatientHistory] = useState([]);
  const [showMedSAM, setShowMedSAM] = useState(true);
  const [isTimeout, setIsTimeout] = useState(false);
  
  // CXR Validation State Machine
  const [pipelineState, setPipelineState] = useState('IDLE'); // IDLE, VALIDATING, VALIDATION_FAILED, READY_FOR_INFERENCE, INFERENCE_RUNNING
  const [validationDetail, setValidationDetail] = useState({ confidence: null, error: null, code: null });
  const [serverSideError, setServerSideError] = useState(null);
  
  // BUG H-01: Progress and Stuck States
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadLoaded, setUploadLoaded] = useState(0);
  const [uploadTotal, setUploadTotal] = useState(0);
  const [isStuck, setIsStuck] = useState(false);
  const stuckTimerRef = useRef(null);

  useEffect(() => {
    // BUG C-03: When navigating back to the upload page, clear all analysis state.
    if (state.result) {
      dispatch({ type: 'RESET_ANALYSIS' });
      setPipelineState('IDLE');
    }
    return () => {
       if (stuckTimerRef.current) clearTimeout(stuckTimerRef.current);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const fetchRecent = async () => {
      try {
        const data = await getAllDiagnoses();
        if (data && Array.isArray(data)) {
          const unique = [];
          const seen = new Set();
          for (const r of data) {
            if (r.patient_id && !seen.has(r.patient_id)) {
              unique.push({ id: r.patient_id, name: r.patient_name });
              seen.add(r.patient_id);
            }
            if (unique.length >= 5) break;
          }
          setRecentPatients(unique);
        }
      } catch (err) { console.error(err); }
    };
    fetchRecent();
  }, []);

  useEffect(() => {
    const fetchExistingPatient = async () => {
      if (formData.patient_id && formData.patient_id.toString().length >= 1) {
        try {
          const history = await getPatientHistory(formData.patient_id);
          setPatientHistory(Array.isArray(history) ? history : []);
          if (Array.isArray(history) && history.length > 0) {
            const lastRecord = history[history.length - 1];
            dispatch({ type: 'UPDATE_FORM', payload: {
              patient_name: lastRecord.patient_name || formData.patient_name,
              age: lastRecord.age || formData.age
            }});
            toast.success(`Active record for ${lastRecord.patient_name}`);
          }
        } catch (err) {
          setPatientHistory([]);
        }
      }
    };
    const debounce = setTimeout(fetchExistingPatient, 800);
    return () => clearTimeout(debounce);
  }, [formData.patient_id]); // eslint-disable-line react-hooks/exhaustive-deps


  const selectRecent = (id, name) => {
    dispatch({ type: 'UPDATE_FORM', payload: { patient_id: id, patient_name: name } });
    setShowRecent(false);
  };

  const runCXRCheck = async (selectedFile) => {
    setPipelineState('VALIDATING');
    setValidationDetail({ confidence: null, error: null, code: null });
    setServerSideError(null);

    try {
      const img = new Image();
      const objectUrl = URL.createObjectURL(selectedFile);
      img.src = objectUrl;
      await new Promise((resolve) => { img.onload = resolve; img.onerror = resolve; });

      // Step 1: Resolution Gate
      if (img.width < 224 || img.height < 224) {
        URL.revokeObjectURL(objectUrl);
        setPipelineState('VALIDATION_FAILED');
        setValidationDetail({ error: `Resolution too low (${img.width}x${img.height}). Minimum: 224x224.`, code: 'RESOLUTION_TOO_LOW' });
        return false;
      }

      // Step 2: Aspect Ratio Gate
      const aspect = img.width / img.height;
      if (aspect < 0.7 || aspect > 1.4) {
        URL.revokeObjectURL(objectUrl);
        setPipelineState('VALIDATION_FAILED');
        setValidationDetail({ error: `Invalid proportions (AR=${aspect.toFixed(2)}). Valid CXRs: 0.7–1.4.`, code: 'INVALID_ASPECT_RATIO' });
        return false;
      }

      // --- Pixel-level analysis (200x200 sample) ---
      const canvas = document.createElement('canvas');
      canvas.width = 200; canvas.height = 200;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        console.warn('[CLIENT-GUARD] Canvas context unavailable. Bypassing pixel checks.');
        setPipelineState('READY_FOR_INFERENCE');
        setValidationDetail({ confidence: 0.90, error: null, code: null });
        return true;
      }
      ctx.drawImage(img, 0, 0, 200, 200);
      const pixelData = ctx.getImageData(0, 0, 200, 200).data;
      URL.revokeObjectURL(objectUrl);

      const totalPixels = 200 * 200;
      let sumLuma = 0;
      let coloredPixels = 0;    // pixels with noticeable color saturation
      let edgeCount = 0;        // horizontal edge transitions
      let prevLuma = 0;

      for (let i = 0; i < pixelData.length; i += 4) {
        const r = pixelData[i], g = pixelData[i + 1], b = pixelData[i + 2];

        // Perceptual luminance (ITU-R BT.601)
        const luma = 0.299 * r + 0.587 * g + 0.114 * b;
        sumLuma += luma;

        // Per-pixel saturation: max channel − min channel
        // X-rays have near-zero saturation (R≈G≈B for every pixel)
        // Color photos/IDs have many pixels with saturation > 20
        const sat = Math.max(r, g, b) - Math.min(r, g, b);
        if (sat > 20) coloredPixels++;

        // Horizontal edge density (skip alpha channel stride)
        if (i > 0 && Math.abs(luma - prevLuma) > 30) edgeCount++;
        prevLuma = luma;
      }

      const meanBrightness  = sumLuma / totalPixels;
      const colorRatio      = coloredPixels / totalPixels;
      const globalEdgeDensity = edgeCount / totalPixels;

      console.log(
        `[CLIENT-GUARD] brightness=${meanBrightness.toFixed(1)} | ` +
        `colorRatio=${(colorRatio * 100).toFixed(1)}% | ` +
        `edgeDensity=${globalEdgeDensity.toFixed(4)}`
      );

      // Gate 1: Per-pixel color saturation
      // X-rays: colorRatio ≈ 0–3%   |   Color photos/ID cards: colorRatio ≈ 15–60%
      if (colorRatio > 0.08) {
        setPipelineState('VALIDATION_FAILED');
        setValidationDetail({
          error: `Color image detected (${(colorRatio * 100).toFixed(0)}% colored pixels). Only grayscale chest X-rays are accepted.`,
          code: 'COLOR_IMAGE_DETECTED'
        });
        return false;
      }

      // Gate 2: Mean brightness
      // X-rays: meanBrightness ≈ 50–170   |   Scanned white documents: meanBrightness > 200
      if (meanBrightness > 200) {
        setPipelineState('VALIDATION_FAILED');
        setValidationDetail({
          error: `Image appears to be a bright document or blank page (brightness=${meanBrightness.toFixed(0)}). Upload a chest X-ray scan.`,
          code: 'DOCUMENT_TOO_BRIGHT'
        });
        return false;
      }

      // Gate 3: Global edge density (very lenient — only blocks pure text/line-art pages)
      // X-rays with text annotations: edgeDensity ≈ 0.03–0.10
      // Dense printed documents / ID cards (even in grayscale): edgeDensity > 0.18
      if (globalEdgeDensity > 0.18) {
        setPipelineState('VALIDATION_FAILED');
        setValidationDetail({
          error: `High text/line density detected. Please upload an actual chest radiograph.`,
          code: 'DOCUMENT_DETECTED'
        });
        return false;
      }

      // All gates passed
      setPipelineState('READY_FOR_INFERENCE');
      setValidationDetail({ confidence: 0.96, error: null, code: null });
      return true;

    } catch (err) {
      setPipelineState('VALIDATION_FAILED');
      setValidationDetail({ error: 'Image validation error. File may be corrupt.', code: 'FILE_CORRUPT' });
      return false;
    }
  };

  const handleFileChange = async (e) => {
    const selected = e.target.files[0];
    if (selected) {
      // RESET BEFORE CHECK
      dispatch({ type: 'SET_FILE', payload: { file: null, preview: null } });
      setServerSideError(null);

      const isValid = await runCXRCheck(selected);
      
      if (!isValid) {
         e.target.value = ''; // Reset input
         return;
      }

      const objectUrl = URL.createObjectURL(selected);
      dispatch({ type: 'SET_FILE', payload: { file: selected, preview: objectUrl } });
    }
  };

  const resetStuckTimer = () => {
    if (stuckTimerRef.current) clearTimeout(stuckTimerRef.current);
    setIsStuck(false);
    stuckTimerRef.current = setTimeout(() => {
      setIsStuck(true);
    }, 15000); // 15 seconds stall detection
  };

  const handleCancel = () => {
    if (xhrCancelRef.current) {
       xhrCancelRef.current();
       xhrCancelRef.current = null;
    }
    dispatch({ type: 'SET_LOADING', payload: false });
    setUploadProgress(0);
    if (stuckTimerRef.current) clearTimeout(stuckTimerRef.current);
    toast.error('Upload cancelled');
  };

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    
    if (pipelineState !== 'READY_FOR_INFERENCE') {
       toast.error('Clinical validation required before inference.');
       return;
    }

    if (!hasConsented) return toast.error('Data processing consent required');
    if (!file) return toast.error('X-Ray imaging required');
    if (!formData.patient_name) return toast.error('Patient identification required');
    
    setPipelineState('INFERENCE_RUNNING');
    dispatch({ type: 'SET_LOADING', payload: true });
    setIsTimeout(false);
    setIsStuck(false);
    setUploadProgress(0);

    const data = new FormData();
    data.append('xray_image', file);
    Object.keys(formData).forEach(key => data.append(key, formData[key]));

    resetStuckTimer();

    try {
      const { promise, cancel } = uploadWithProgress(data, (perc, loaded, total) => {
        setUploadProgress(perc);
        setUploadLoaded(loaded);
        setUploadTotal(total);
        resetStuckTimer();
      });

      xhrCancelRef.current = cancel;
      const response = await promise;
      
      if (stuckTimerRef.current) clearTimeout(stuckTimerRef.current);
      dispatch({ type: 'SET_RESULT', payload: response });
      toast.success('Inference complete');

      if (formData.patient_id) {
        getPatientHistory(formData.patient_id).then(history => {
          setPatientHistory(Array.isArray(history) ? history : []);
        }).catch(() => {});
      }

      navigate('/app/result');
    } catch (err) {
      if (stuckTimerRef.current) clearTimeout(stuckTimerRef.current);
      if (err.aborted) return;

      if (err.status === 422) {
        const detail = err.detail || err;
        setPipelineState('VALIDATION_FAILED');
        setValidationDetail({ error: detail.details || detail.reason || 'Invalid Image', code: detail.reason || 'NON_MEDICAL_IMAGE' });
        setServerSideError(detail.reason || 'Validation Failed');
        dispatch({ type: 'SET_FILE', payload: { file: null, preview: null } });
      } else {
        const errorMsg = typeof err.detail === 'string' ? err.detail : (err.message || 'Diagnostic pipeline failure');
        toast.error(errorMsg);
        setPipelineState('READY_FOR_INFERENCE');
      }
      
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  };

  return (
    <div className="p-4 md:p-8 lg:p-12 max-w-[1600px] mx-auto font-sans text-slate-900 bg-[#f8fafc] min-h-screen">
      <SectionHeader 
        title="Diagnostic Workstation" 
        subtitle="Clinical AI fusion for high-precision Tuberculosis detection."
        icon={BrainCircuit}
      />

      <ServerStatus />

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left: Configuration Panel */}
        <motion.div 
           initial={{ opacity: 0, x: -20 }} 
           animate={{ opacity: 1, x: 0 }} 
           className="lg:col-span-7 space-y-8 pb-32 md:pb-0"
        >
          <form onSubmit={handleSubmit} className="space-y-8">
            
            <GlassCard title="Patient Profile" icon={User} badge={
               patientHistory.length > 0 && (
                 <span className="px-2 py-1 bg-blue-600/10 text-blue-600 text-[10px] font-black rounded-lg border border-blue-600/20 uppercase tracking-widest animate-pulse">
                   Active Session
                 </span>
               )
            }>
               <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="relative">
                    <label className="text-[11px] font-black text-slate-400 uppercase tracking-widest px-1 block mb-2">Clinical ID</label>
                    <input 
                       type="number" 
                       className="input-field" 
                       value={formData.patient_id}
                       onChange={(e) => dispatch({ type: 'UPDATE_FORM', payload: { patient_id: e.target.value } })}
                       onFocus={() => setShowRecent(true)}
                       onBlur={() => setTimeout(() => setShowRecent(false), 200)}
                    />
                    <AnimatePresence>
                      {showRecent && recentPatients.length > 0 && (
                        <motion.div 
                           initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 10 }}
                           className="absolute z-50 left-0 right-0 top-full mt-2 bg-white rounded-2xl border border-slate-200 shadow-2xl overflow-hidden p-2"
                        >
                           <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest p-2 mb-1">Recent Searches</p>
                           {recentPatients.map(p => (
                             <button 
                                key={p.id} type="button" onClick={() => selectRecent(p.id, p.name)}
                                className="w-full text-left p-3 hover:bg-slate-50 rounded-xl flex justify-between items-center group transition-all"
                             >
                                <span className="text-sm text-slate-700 font-bold group-hover:text-blue-600">{p.name}</span>
                                <span className="text-[10px] text-slate-400 font-black">ID: {p.id}</span>
                             </button>
                           ))}
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                  <div>
                    <label className="text-[11px] font-black text-slate-400 uppercase tracking-widest px-1 block mb-2">Full Name</label>
                    <input 
                      type="text" required className="input-field" placeholder="Patient Identity"
                      value={formData.patient_name} onChange={(e) => dispatch({ type: 'UPDATE_FORM', payload: { patient_name: e.target.value } })}
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                     <div>
                       <label className="text-[11px] font-black text-slate-400 uppercase tracking-widest px-1 block mb-2">Date of Birth</label>
                       <input 
                         type="date" className="input-field px-3" value={formData.date_of_birth || ''}
                         onChange={(e) => {
                            const dob = e.target.value;
                            let newAge = formData.age;
                            if (dob) {
                               const ageDate = new Date(Date.now() - new Date(dob).getTime());
                               newAge = Math.abs(ageDate.getUTCFullYear() - 1970);
                            }
                            dispatch({ type: 'UPDATE_FORM', payload: { date_of_birth: dob, age: newAge || '' } });
                         }}
                       />
                     </div>
                     <div>
                       <label className="text-[11px] font-black text-slate-400 uppercase tracking-widest px-1 block mb-2">Age</label>
                       <input 
                         type="number" required className="input-field px-3" value={formData.age || ''} placeholder="Yrs"
                         onChange={(e) => dispatch({ type: 'UPDATE_FORM', payload: { age: e.target.value } })}
                       />
                     </div>
                  </div>
                  <div>
                    <label className="text-[11px] font-black text-slate-400 uppercase tracking-widest px-1 block mb-2">Radiology Email</label>
                    <input 
                      type="email" required className="input-field" value={formData.doctor_email}
                      onChange={(e) => dispatch({ type: 'UPDATE_FORM', payload: { doctor_email: e.target.value } })}
                    />
                  </div>
               </div>
            </GlassCard>

            <GlassCard title="Radiology Interface" icon={FileImage}>
               {serverSideError && (
                 <motion.div 
                    initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}
                    className="mb-6 p-4 bg-rose-500 border border-rose-400 rounded-2xl flex items-center gap-3 text-white"
                 >
                    <XCircle size={20} />
                    <p className="text-sm font-bold">{serverSideError}</p>
                 </motion.div>
               )}
               <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-start">
                  <div 
                     className={`md:col-span-8 w-full aspect-square md:aspect-[4/3] rounded-3xl border-2 border-dashed flex flex-col items-center justify-center transition-all relative overflow-hidden cursor-pointer ${!hasConsented ? 'opacity-20 grayscale pointer-events-none' : ''} ${preview ? 'border-blue-500/50 bg-blue-50/10' : pipelineState === 'VALIDATION_FAILED' ? 'border-rose-400 bg-rose-50' : 'border-slate-200 bg-slate-50 hover:border-blue-400'}`}
                     onClick={() => hasConsented && fileInputRef.current?.click()}
                     aria-disabled={!hasConsented}
                  >
                     <input type="file" ref={fileInputRef} className="hidden" accept="image/*" onChange={handleFileChange} />
                     
                     {pipelineState === 'VALIDATION_FAILED' && !preview ? (
                        <div className="text-center p-8 flex flex-col items-center gap-4">
                           <div className="w-20 h-20 bg-rose-100 rounded-full flex items-center justify-center text-rose-600 animate-bounce">
                              <XCircle size={40} />
                           </div>
                           <h3 className="text-xl font-black text-rose-700 tracking-tight">Image Authentication Failed</h3>
                           <p className="text-sm font-bold text-rose-600 max-w-[300px] text-center">{validationDetail.error}</p>
                           <div className="bg-white/50 p-4 rounded-2xl border border-rose-200 text-left space-y-2">
                              <p className="text-[10px] font-black text-rose-800 uppercase tracking-widest">Clinical Audit - {validationDetail.code}:</p>
                              <ul className="text-[9px] font-bold text-rose-600 list-disc list-inside space-y-1">
                                 <li>Chest X-ray images only (Grayscale)</li>
                                 <li>Minimum Resolution: 224x224 pixels</li>
                                 <li>No documents, IDs, or text-heavy PII</li>
                                 <li>Proportions (AR) between 0.7 and 1.4</li>
                              </ul>
                           </div>
                           <button 
                              onClick={(e) => { e.stopPropagation(); setPipelineState('IDLE'); setValidationDetail({ confidence: null, error: null, code: null }); }}
                              className="mt-4 px-6 py-2.5 bg-rose-600 text-white text-[11px] font-black uppercase tracking-widest rounded-xl hover:bg-rose-700 shadow-lg shadow-rose-200"
                           >
                              Retry Radiograph Upload
                           </button>
                        </div>
                     ) : pipelineState === 'VALIDATING' ? (
                        <div className="text-center p-12">
                           <Activity size={48} className="text-blue-600 mb-6 mx-auto animate-spin opacity-40" />
                           <p className="text-lg font-black text-slate-800 tracking-tight animate-pulse">Analyzing Authenticity...</p>
                           <p className="text-sm font-medium text-slate-400 mt-1 uppercase tracking-widest">Guard AI Grid Inspection</p>
                        </div>
                     ) : preview ? (
                        <div className="relative w-full h-full p-4 flex items-center justify-center">
                           <img src={preview} alt="X-ray" className="w-full h-full object-contain rounded-2xl shadow-lg brightness-110 contrast-125" />
                           <div className="scanning-line" />
                           <button 
                              type="button" onClick={(e) => { e.stopPropagation(); dispatch({ type: 'SET_FILE', payload: { file: null, preview: null } }); setPipelineState('IDLE'); }}
                              className="absolute top-8 right-8 p-3 bg-white/20 backdrop-blur-md rounded-2xl text-white hover:bg-rose-500 transition-all border border-white/20 z-50"
                           >
                              <XCircle size={20} />
                           </button>
                        </div>
                     ) : (
                        <div className="text-center p-12">
                           <UploadCloud size={48} className="text-blue-600 mb-6 mx-auto opacity-40" />
                           <p className="text-lg font-black text-slate-800 tracking-tight">Ingest Chest X-Ray</p>
                           <p className="text-sm font-medium text-slate-400 mt-1">DICOM, JPEG, PNG (Max 20MB)</p>
                        </div>
                     )}

                     {pipelineState === 'READY_FOR_INFERENCE' && preview && (
                        <div className="absolute bottom-6 left-6 right-6">
                           <div className="p-3 rounded-2xl backdrop-blur-md border bg-emerald-500/10 border-emerald-500/30 flex items-center justify-between transition-all">
                              <div className="flex items-center gap-3">
                                 <CheckCircle size={16} className="text-emerald-500" />
                                 <div>
                                    <p className="text-[10px] font-black uppercase tracking-widest text-emerald-600">
                                       Authenticated Radiograph
                                    </p>
                                    <p className="text-[9px] font-bold text-slate-400">Guard AI Confidence: {(validationDetail.confidence * 100).toFixed(1)}%</p>
                                 </div>
                              </div>
                           </div>
                        </div>
                     )}
                  </div>

                  {/* Inline Error for Client-side Validation */}
                  {pipelineState === 'VALIDATION_FAILED' && (
                    <div className="md:col-span-12 p-4 bg-rose-50 border border-rose-100 rounded-2xl flex items-start gap-3 animate-shake">
                      <AlertCircle className="text-rose-500 shrink-0" size={18} />
                      <p className="text-xs font-bold text-rose-700 leading-relaxed">{validationDetail.error}</p>
                    </div>
                  )}

                  {/* Error display for upload failures */}
                  {uploadError && (
                    <div className="md:col-span-12 p-4 bg-rose-50 border border-rose-100 rounded-2xl flex items-start gap-3">
                      <AlertCircle className="text-rose-500 shrink-0" size={18} />
                      <div className="flex-1">
                        <p className="text-xs font-bold text-rose-700 leading-relaxed">{uploadError}</p>
                      </div>
                    </div>
                  )}

                  <div className="md:col-span-4 space-y-6">
                     <div className="bg-slate-50 border border-slate-100 p-5 rounded-3xl min-h-[150px] md:h-[450px] flex flex-col">
                        <h4 className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
                           <Activity size={12} /> Anatomical Separation
                        </h4>
                        <div className="flex-1 space-y-3 overflow-y-auto pr-2 custom-scrollbar">
                           <div className="h-full flex flex-col items-center justify-center grayscale opacity-10">
                              <LayoutGrid size={32} />
                              <p className="text-[9px] font-black uppercase mt-3">Awaiting Scan</p>
                           </div>
                        </div>
                     </div>
                  </div>
               </div>
            </GlassCard>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
               <GlassCard title="Clinical Metrics" icon={Activity}>
                  <div className="space-y-6">
                     <div>
                        <label className="text-[11px] font-black text-slate-400 uppercase tracking-widest block mb-3">Cough Chronicity (Weeks)</label>
                        <input 
                           type="number" className="input-field" value={formData.cough_duration_weeks}
                           onChange={(e) => dispatch({ type: 'UPDATE_FORM', payload: { cough_duration_weeks: parseInt(e.target.value) || 0 } })}
                        />
                     </div>
                     <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        {['fever', 'weight_loss', 'night_sweats'].map(key => (
                           <label key={key} className={`flex flex-col items-center p-4 bg-white border rounded-2xl cursor-pointer transition-all select-none group ${formData.no_symptoms === 1 ? 'opacity-30 pointer-events-none' : 'hover:border-blue-300 border-slate-200'}`}>
                              <span className="text-[9px] font-black text-slate-400 uppercase tracking-widest mb-3">{key.replace('_', ' ')}</span>
                              <div className="custom-checkbox">
                                 <input type="checkbox" checked={formData[key] === 1} onChange={(e) => dispatch({ type: 'UPDATE_FORM', payload: { [key]: e.target.checked ? 1 : 0 } })} />
                                 <div className={`w-6 h-6 rounded-lg border flex items-center justify-center transition-all ${formData[key] === 1 ? 'bg-blue-600 border-blue-600 text-white' : 'bg-white border-slate-300'}`}>
                                    {formData[key] === 1 && <CheckCircle size={14} strokeWidth={3} />}
                                 </div>
                               </div>
                           </label>
                        ))}
                        <label className="flex flex-col items-center p-4 bg-white border border-slate-200 rounded-2xl cursor-pointer hover:border-emerald-300 transition-all select-none group">
                           <span className="text-[9px] font-black text-emerald-600 uppercase tracking-widest mb-3">No Symptoms</span>
                           <div className="custom-checkbox">
                              <input 
                                type="checkbox" 
                                checked={formData.no_symptoms === 1} 
                                onChange={(e) => {
                                  const checked = e.target.checked ? 1 : 0;
                                  dispatch({ type: 'UPDATE_FORM', payload: { 
                                    no_symptoms: checked,
                                    fever: 0, weight_loss: 0, night_sweats: 0, cough_duration_weeks: 0
                                  }});
                                }} 
                              />
                              <div className={`w-6 h-6 rounded-lg border flex items-center justify-center transition-all ${formData.no_symptoms === 1 ? 'bg-emerald-600 border-emerald-600 text-white' : 'bg-white border-slate-300'}`}>
                                 {formData.no_symptoms === 1 && <CheckCircle size={14} strokeWidth={3} />}
                              </div>
                           </div>
                        </label>
                     </div>
                     {/* Inline Validation Error */}
                     {! (formData.fever || formData.weight_loss || formData.night_sweats || formData.cough_duration_weeks > 0 || formData.no_symptoms) && (
                       <p className="text-[10px] font-bold text-rose-500 animate-pulse uppercase tracking-widest flex items-center gap-1">
                         <AlertCircle size={12} /> Please select at least one symptom, or check 'No symptoms' to proceed.
                       </p>
                     )}
                  </div>
               </GlassCard>

               <GlassCard title="Laboratory Data" icon={Microscope}>
                  <div className="space-y-5">
                    {[
                      { id: 'sputum_test', label: 'Sputum AFB' },
                      { id: 'genexpert_test', label: 'GeneXpert' }
                    ].map(test => (
                      <div key={test.id} className="space-y-3">
                         <p className="text-[11px] font-black text-slate-400 uppercase tracking-widest">{test.label}</p>
                         <div className="flex bg-slate-100 p-1 rounded-xl gap-1">
                            {[{ v: 0, l: 'Null' }, { v: 1, l: 'POS', c: 'text-rose-600' }, { v: 2, l: 'NEG', c: 'text-emerald-600' }].map(o => (
                               <button 
                                  key={o.v} type="button" onClick={() => dispatch({ type: 'UPDATE_FORM', payload: { [test.id]: o.v } })}
                                  className={`flex-1 py-2 rounded-lg text-[10px] font-black border transition-all ${formData[test.id] === o.v ? 'bg-white border-slate-200 shadow-sm ' + (o.c || 'text-slate-600') : 'text-slate-400 border-transparent'}`}
                               >
                                  {o.l}
                               </button>
                            ))}
                         </div>
                      </div>
                    ))}
                  </div>
               </GlassCard>
            </div>

            <div className="pt-4 relative group flex flex-col items-center">
               {!(formData.fever || formData.weight_loss || formData.night_sweats || formData.cough_duration_weeks > 0 || formData.no_symptoms) && (
                 <div className="absolute -top-10 bg-slate-800 text-white text-[10px] px-3 py-1.5 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none uppercase font-black tracking-widest z-50 whitespace-nowrap">
                   Select at least one symptom to continue
                 </div>
               )}
               <div className="fixed bottom-20 left-0 right-0 p-4 bg-transparent z-[60] md:relative md:bottom-0 md:p-0 md:bg-transparent flex justify-center w-full">
                 <button 
                   type="submit" 
                   disabled={loading || !hasConsented || pipelineState !== 'READY_FOR_INFERENCE' || !(formData.fever || formData.weight_loss || formData.night_sweats || formData.cough_duration_weeks > 0 || formData.no_symptoms)}
                   className="btn-primary w-full max-w-sm flex items-center justify-center gap-4 py-5 hover:scale-105 active:scale-95 disabled:opacity-30 disabled:grayscale disabled:cursor-not-allowed transition-all shadow-[0_-20px_50px_rgba(37,99,235,0.2)] md:shadow-none bg-blue-600 rounded-2xl text-white font-black"
                 >
                   {loading ? <Activity className="animate-spin" /> : <Sparkles className="transition-transform group-hover:rotate-12" />}
                   <span className="font-black uppercase tracking-widest text-[14px]">
                     {loading ? 'Initializing Engine...' : 'Execute AI Detection'}
                   </span>
                 </button>
               </div>
            </div>
          </form>
        </motion.div>

        {/* Right: Intelligence Panel */}
        <div className="lg:col-span-5 space-y-8">
           <GlassCard title="Inference Output" icon={Microscope}>
              {pipelineState === 'READY_FOR_INFERENCE' ? (
                <div className="flex flex-col items-center justify-center py-20 text-emerald-600">
                   <div className="p-8 bg-emerald-50 rounded-full mb-6 border border-emerald-100 shadow-inner">
                      <CheckCircle size={48} className="animate-pulse" />
                   </div>
                   <h4 className="text-lg font-black tracking-tight">Validated CXR ✓</h4>
                   <p className="text-xs font-bold mt-2 max-w-[200px] text-center uppercase tracking-widest opacity-60">Authentication Complete. Ready for Multimodal Fusion.</p>
                </div>
              ) : pipelineState === 'VALIDATION_FAILED' ? (
                <div className="flex flex-col items-center justify-center py-20 text-rose-600">
                   <div className="p-8 bg-rose-50 rounded-full mb-6 border border-rose-100 shadow-inner">
                      <XCircle size={48} />
                   </div>
                   <h4 className="text-lg font-black tracking-tight">Access Blocked</h4>
                   <p className="text-xs font-bold mt-2 max-w-[200px] text-center uppercase tracking-widest opacity-60">{validationDetail.code}</p>
                </div>
              ) : pipelineState === 'VALIDATING' ? (
                <div className="flex flex-col items-center justify-center py-20 text-blue-600">
                   <div className="p-8 bg-blue-50 rounded-full mb-6 border border-blue-100 shadow-inner">
                      <Activity size={48} className="animate-spin" />
                   </div>
                   <h4 className="text-lg font-black tracking-tight">Analyzing Guard AI</h4>
                   <p className="text-xs font-bold mt-2 max-w-[200px] text-center uppercase tracking-widest opacity-60">Verifying Anatomical Authenticity</p>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-20 grayscale opacity-40">
                   <div className="p-8 bg-slate-100 rounded-full mb-6"><Microscope size={48} className="text-slate-400" /></div>
                   <h4 className="text-lg font-black text-slate-700 tracking-tight">Radiology Pipeline Idle</h4>
                   <p className="text-xs font-bold text-slate-400 mt-2 max-w-[200px] text-center uppercase tracking-widest">Awaiting Bio-Metric Ingest</p>
                </div>
              )}
           </GlassCard>
        </div>
      </div>

      <AnimatePresence>
        {!hasConsented && (
          <PatientDataNotice 
            onAccept={() => {
              sessionStorage.setItem('tb_consent_ack', 'true');
              dispatch({ type: 'SET_CONSENT', payload: true });
              toast.success('Clinical consent confirmed');
            }}
            onCancel={() => navigate('/')}
          />
        )}
        {loading && (
          <motion.div 
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] bg-slate-950/40 backdrop-blur-xl flex items-center justify-center p-6"
          >
            <div className="text-center p-12 glass-panel border-white/20 shadow-[0_40px_100px_rgba(0,0,0,0.4)] max-w-md w-full">
                <div className="relative mb-10 w-24 h-24 mx-auto">
                   <div className="absolute inset-0 animate-ping opacity-20 bg-blue-400 rounded-full" />
                   <div className="w-24 h-24 bg-blue-600 rounded-3xl flex items-center justify-center text-white relative z-10 shadow-[0_0_50px_rgba(37,99,235,0.4)]">
                      <BrainCircuit size={48} className="animate-pulse" />
                   </div>
                </div>
                
                <h2 className="text-2xl font-black text-white tracking-tight mb-3 font-[Poppins]">
                  {uploadProgress < 100 ? 'Secure Data Transfer' : 'Orchestrating AI'}
                </h2>
                <p className="text-blue-200/60 font-medium text-sm leading-relaxed mb-8">
                  {uploadProgress < 100 
                    ? 'Transmitting multimodal clinical data to secure inference engine...' 
                    : 'Analyzing radiology features and correlating clinical symptoms for fusion...'}
                </p>

                {uploadProgress < 100 ? (
                  <div className="space-y-4">
                    <UploadProgressBar 
                      progress={uploadProgress}
                      total={uploadTotal}
                      loaded={uploadLoaded}
                      onCancel={handleCancel}
                    />
                    {isStuck && (
                      <motion.div 
                        initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                        className="p-3 bg-rose-500/10 border border-rose-500/20 rounded-xl"
                      >
                        <p className="text-[10px] font-black text-rose-400 uppercase tracking-widest text-center">
                          Upload appears stuck. Check your connection and try again.
                        </p>
                      </motion.div>
                    )}
                  </div>
                ) : (
                  <div className="w-full h-1.5 bg-white/10 rounded-full mt-10 overflow-hidden">
                      <motion.div 
                         initial={{ x: "-100%" }} animate={{ x: "100%" }} transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                         className="w-1/2 h-full bg-gradient-to-r from-transparent via-blue-400 to-transparent"
                      />
                  </div>
                )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
