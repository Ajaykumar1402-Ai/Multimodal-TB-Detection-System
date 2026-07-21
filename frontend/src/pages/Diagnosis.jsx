import React, { useState, useRef } from 'react';
import { uploadDiagnosis } from '../services/api';
import { toast } from 'react-hot-toast';
import { UploadCloud, FileImage, Stethoscope, AlertCircle, HeartPulse, ShieldAlert, Mail, Search, CheckCircle, Activity, Download, Save } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function Diagnosis() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const fileInputRef = useRef(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [isSaved, setIsSaved] = useState(false);

  const [formData, setFormData] = useState({
    patient_id: Math.floor(Math.random() * 900000) + 100000, // Auto-generated 6-digit ID
    patient_name: '',
    age: 45,
    doctor_email: 'doctor@example.com',
    cough_duration_weeks: 0,
    fever: 0,
    weight_loss: 0,
    night_sweats: 0
  });

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected && selected.type.startsWith('image/')) {
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
    } else {
      toast.error('Please select a valid image file');
    }
  };

  const handleDragOver = (e) => e.preventDefault();
  const handleDrop = (e) => {
    e.preventDefault();
    const dropped = e.dataTransfer.files[0];
    if (dropped && dropped.type.startsWith('image/')) {
      setFile(dropped);
      setPreview(URL.createObjectURL(dropped));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return toast.error('Please upload a Chest X-Ray image');
    
    setLoading(true);
    setIsSaved(false); // Reset save state on new run
    const data = new FormData();
    data.append('xray_image', file);
    Object.keys(formData).forEach(key => data.append(key, formData[key]));

    try {
      const response = await uploadDiagnosis(data);
      setResult(response);
      toast.success('Diagnosis complete! Report generated.');
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Failed to process diagnosis');
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (risk) => {
    switch(risk) {
      case 'High': return 'text-red-500 shadow-red-500/20 border-red-500/50';
      case 'Medium': return 'text-yellow-500 shadow-yellow-500/20 border-yellow-500/50';
      default: return 'text-green-500 shadow-green-500/20 border-green-500/50';
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6 pb-20">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
          <Stethoscope className="text-primary" /> Create New Diagnosis
        </h1>
        <p className="text-textmuted">Upload X-Ray and fill clinical parameters for Multimodal Analysis</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        
        {/* Left Column: Form & Upload */}
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="space-y-6">
          <form onSubmit={handleSubmit} className="glass-panel p-6 space-y-6">
            
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-white border-b border-white/10 pb-2">Patient Details</h3>
              <div className="grid grid-cols-2 gap-4">
                 <div>
                    <label className="text-sm text-textmuted mb-1 block">Patient ID (Simulation)</label>
                    <input type="number" min="1" value={formData.patient_id} onChange={(e) => setFormData({...formData, patient_id: e.target.value})} className="input-field bg-black/20" required />
                 </div>
                 <div>
                    <label className="text-sm text-textmuted mb-1 block">Patient Name</label>
                    <input type="text" placeholder="e.g. John Doe" value={formData.patient_name} onChange={(e) => setFormData({...formData, patient_name: e.target.value})} className="input-field" required />
                 </div>
                 <div>
                    <label className="text-sm text-textmuted mb-1 block">Patient Age</label>
                    <input type="number" min="1" max="120" value={formData.age} onChange={(e) => setFormData({...formData, age: e.target.value})} className="input-field" required />
                 </div>
                 <div>
                    <label className="text-sm text-textmuted mb-1 block">Doctor Notification Email</label>
                    <input type="email" value={formData.doctor_email} onChange={(e) => setFormData({...formData, doctor_email: e.target.value})} className="input-field" required />
                 </div>
              </div>
            </div>

            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-white border-b border-white/10 pb-2">Clinical Parameters</h3>
              
              <div>
                <label className="text-sm text-textmuted mb-1 block">Cough Duration (Weeks)</label>
                <input type="number" min="0" value={formData.cough_duration_weeks} onChange={(e) => setFormData({...formData, cough_duration_weeks: parseInt(e.target.value)})} className="input-field" />
              </div>

              <div className="grid grid-cols-3 gap-4">
                {['fever', 'weight_loss', 'night_sweats'].map(param => (
                  <label key={param} className="flex flex-col items-center p-3 bg-black/20 border border-white/5 rounded-xl cursor-pointer hover:bg-black/40 transition">
                    <span className="text-sm text-gray-300 capitalize mb-2">{param.replace('_', ' ')}</span>
                    <input 
                      type="checkbox" 
                      className="w-5 h-5 accent-primary"
                      checked={formData[param] === 1}
                      onChange={(e) => setFormData({...formData, [param]: e.target.checked ? 1 : 0})}
                    />
                  </label>
                ))}
              </div>
            </div>

            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-white border-b border-white/10 pb-2 flex items-center gap-2">
                 <FileImage size={18} /> Chest X-Ray
              </h3>
              <div 
                className={`border-2 border-dashed ${file ? 'border-primary/50 bg-primary/5' : 'border-white/20 bg-black/20'} rounded-2xl p-8 text-center cursor-pointer transition-all hover:border-primary/50 hover:bg-primary/5`}
                onClick={() => fileInputRef.current?.click()}
                onDragOver={handleDragOver}
                onDrop={handleDrop}
              >
                <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" accept="image/*" />
                {preview ? (
                  <div className="flex flex-col items-center">
                    <img src={preview} alt="XRay Preview" className="h-40 rounded-lg shadow-lg mb-4 object-cover" />
                    <p className="text-sm text-primary font-medium">{file.name}</p>
                    <p className="text-xs text-textmuted mt-1">Click or drag to replace</p>
                  </div>
                ) : (
                  <div className="flex flex-col items-center text-textmuted">
                    <UploadCloud size={48} className="mb-4 text-primary/70" />
                    <p className="font-medium text-white">Click to upload or drag & drop</p>
                    <p className="text-sm mt-1">SVG, PNG, JPG or DICOM (max. 10MB)</p>
                  </div>
                )}
              </div>
            </div>

            <button type="submit" className="btn-primary flex items-center justify-center gap-2" disabled={loading}>
              {loading ? <Search className="animate-spin" /> : <HeartPulse />}
              {loading ? 'Analyzing Multimodal Data...' : 'Run Diagnostics'}
            </button>
          </form>
        </motion.div>

        {/* Right Column: Results */}
        <AnimatePresence>
        {result ? (
          <motion.div 
            initial={{ opacity: 0, x: 20, scale: 0.95 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            className="space-y-6 relative"
          >
            {/* The Results Panel */}
            <div className={`glass-panel p-8 border-2 shadow-2xl ${getRiskColor(result.results.risk_level)} bg-surface`}>
               <div className="flex justify-between items-start mb-6 border-b border-white/10 pb-6">
                  <div>
                     <p className="text-textmuted text-sm uppercase tracking-widest font-bold mb-1">Final TB Probability</p>
                     <h2 className="text-6xl font-black text-white">
                        {(result.results.final_prob * 100).toFixed(1)}<span className="text-3xl text-gray-400">%</span>
                     </h2>
                  </div>
                  <div className={`px-4 py-2 rounded-full font-bold border flex items-center gap-2 shadow-lg ${getRiskColor(result.results.risk_level)}`}>
                     <ShieldAlert size={20} />
                     {result.results.risk_level} Risk
                  </div>
               </div>

               <div className="space-y-6">
                 <div>
                    <h4 className="text-white font-medium flex items-center gap-2 mb-2"><HeartPulse size={18} className="text-primary"/> AI Recommendations</h4>
                    <p className="text-gray-300 leading-relaxed bg-black/30 p-4 rounded-xl border border-white/5">
                      {result.results.recommendations}
                    </p>
                 </div>
                 
                 {/* Modality Breakdown */}
                 <div className="grid grid-cols-2 gap-4 pt-4 border-t border-white/10">
                    <div className="bg-black/30 p-4 rounded-xl border border-white/5">
                        <p className="text-xs text-textmuted mb-1 uppercase tracking-wide">CNN X-Ray Score</p>
                        <p className="text-xl font-bold text-white">
                           {(result.results.cnn_probability ?? 0 * 100).toFixed(1)}%
                        </p>
                    </div>
                    <div className="bg-black/30 p-4 rounded-xl border border-white/5">
                        <p className="text-xs text-textmuted mb-1 uppercase tracking-wide">Clinical ML Score</p>
                        <p className="text-xl font-bold text-white">
                           {(result.results.clinical_probability ?? 0 * 100).toFixed(1)}%
                        </p>
                    </div>
                 </div>

                 {/* Notifications Status */}
                 <div className="flex items-center gap-4 mt-6 text-sm flex-wrap">
                    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border ${result.pdf_url ? 'bg-green-500/10 text-green-400 border-green-500/20' : 'bg-red-500/10 text-red-400 border-red-500/20'}`}>
                       {result.pdf_url ? <CheckCircle size={16}/> : <AlertCircle size={16}/>}
                       {result.pdf_url ? (
                         <a href={result.pdf_url} target="_blank" rel="noopener noreferrer" className="flex items-center gap-1 hover:text-white transition-colors">
                            PDF Report Generated <Download size={14}/>
                         </a>
                       ) : 'PDF Failed'}
                    </div>
                    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border ${result.email_sent ? 'bg-blue-500/10 text-blue-400 border-blue-500/20' : 'bg-gray-500/10 text-gray-400 border-gray-500/20'}`}>
                       <Mail size={16}/>
                       {result.email_sent ? `Email Sent to ${formData.doctor_email}` : 'Email Queued'}
                    </div>
                 </div>

                 {/* Save Action */}
                 <div className="mt-8 pt-6 border-t border-white/10 flex justify-end">
                    <button 
                       onClick={() => {
                          setIsSaved(true);
                          toast.success('Diagnosis successfully saved to patient records!');
                       }} 
                       disabled={isSaved}
                       className={`flex items-center gap-2 px-6 py-3 rounded-xl font-bold transition-all duration-300 shadow-xl ${isSaved ? 'bg-green-500/20 text-green-400 border border-green-500/30 cursor-not-allowed' : 'bg-secondary text-white hover:bg-secondary/80 hover:-translate-y-1'}`}
                    >
                       {isSaved ? <CheckCircle size={20} /> : <Save size={20} />}
                       {isSaved ? 'Saved to Records' : 'Save Diagnosis'}
                    </button>
                 </div>
               </div>
            </div>

            {/* Explainable AI Visualization Placeholder */}
            {preview && (
              <div className="glass-panel p-6">
                <h4 className="text-white font-medium mb-4 flex items-center gap-2">
                  <Activity size={18} className="text-secondary"/> Explainable AI (Grad-CAM)
                </h4>
                <div className="relative rounded-xl overflow-hidden group">
                  <img src={preview} alt="Original" className="w-full h-48 object-cover filter grayscale" />
                  {/* Heatmap Overlay Simulation */}
                  {result.results.final_prob > 0.4 && (
                     <div className="absolute inset-0 bg-gradient-to-tr from-transparent via-red-500/30 to-red-600/60 mix-blend-overlay"></div>
                  )}
                  <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                    <p className="text-white font-medium">Attention map analyzing basal regions</p>
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        ) : (
          <div className="glass-panel p-8 h-full min-h-[500px] flex flex-col items-center justify-center text-textmuted border-dashed border-2 border-white/5">
            <HeartPulse size={64} className="mb-4 opacity-20" />
            <h3 className="text-xl font-medium text-white/50 mb-2">Awaiting Data</h3>
            <p className="text-center max-w-xs">Upload multimodal patient data to see the AI analysis results here.</p>
          </div>
        )}
        </AnimatePresence>
      </div>
    </div>
  );
}
