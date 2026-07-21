import React, { useState } from 'react';
import { ShieldCheck, AlertTriangle, Lock, FileText, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const PatientDataNotice = ({ onAccept, onCancel }) => {
  const [hasObtainedConsent, setHasObtainedConsent] = useState(false);

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-2xl">
      <motion.div 
        initial={{ opacity: 0, scale: 0.9, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        className="bg-white rounded-[2.5rem] shadow-[0_40px_100px_rgba(0,0,0,0.5)] w-full max-w-2xl overflow-hidden border border-slate-200"
      >
        <div className="p-8 md:p-12">
          <div className="flex items-center gap-4 mb-8">
            <div className="p-3 bg-blue-50 text-blue-600 rounded-2xl">
              <ShieldCheck size={32} />
            </div>
            <div>
              <h2 className="text-2xl font-black text-slate-900 tracking-tight uppercase">Patient Data Notice</h2>
              <p className="text-[10px] font-black text-blue-600 uppercase tracking-widest">Regulatory Compliance: GDPR • HIPAA • DPDP 2023</p>
            </div>
          </div>

          <div className="space-y-6 max-h-[40vh] overflow-y-auto pr-4 custom-scrollbar text-slate-600">
            <section>
              <h3 className="text-xs font-black text-slate-900 uppercase tracking-widest mb-2 flex items-center gap-2">
                <FileText size={14} className="text-blue-500" /> Data Collection & Purpose
              </h3>
              <p className="text-sm leading-relaxed">
                TB-Vision Pro collects clinical metadata (age, symptoms, lab results) and chest radiographs solely for the purpose of Tuberculosis screening using AI. This data is processed to generate diagnostic insights and clinical reports.
              </p>
            </section>

            <section>
              <h3 className="text-xs font-black text-slate-900 uppercase tracking-widest mb-2 flex items-center gap-2">
                <Lock size={14} className="text-blue-500" /> Processing & Security
              </h3>
              <p className="text-sm leading-relaxed">
                Data is processed on secure cloud servers (Render/AWS). All transmissions are encrypted via SSL/TLS. <strong>Retention Policy:</strong> Diagnostic images are cached for processing and not stored permanently after the analysis session is closed.
              </p>
            </section>

            <section>
              <h3 className="text-xs font-black text-slate-900 uppercase tracking-widest mb-2 flex items-center gap-2">
                <ShieldCheck size={14} className="text-blue-500" /> Access & Rights
              </h3>
              <p className="text-sm leading-relaxed">
                Only authorized medical professionals who initiate the scan have access to the results. Under GDPR and DPDP 2023, you have the right to access, rectify, or withdraw data before final submission.
              </p>
            </section>
          </div>

          <div className="mt-10 p-6 bg-slate-50 rounded-3xl border border-slate-100">
            <label className="flex items-start gap-4 cursor-pointer group">
              <div className="relative flex items-center mt-1">
                <input 
                  type="checkbox" 
                  className="peer h-5 w-5 cursor-pointer appearance-none rounded-md border-2 border-slate-300 transition-all checked:bg-blue-600 checked:border-blue-600"
                  checked={hasObtainedConsent}
                  onChange={(e) => setHasObtainedConsent(e.target.checked)}
                />
                <ShieldCheck className="absolute h-3 w-3 text-white opacity-0 peer-checked:opacity-100 left-1 pointer-events-none" />
              </div>
              <span className="text-sm font-bold text-slate-700 leading-tight">
                I confirm I have obtained explicit patient consent to upload this radiograph and clinical data for AI processing.
              </span>
            </label>
          </div>

          <div className="mt-8 flex flex-col md:flex-row gap-4">
            <button 
              onClick={onAccept}
              disabled={!hasObtainedConsent}
              className="flex-1 py-4 px-6 bg-slate-900 text-white rounded-2xl font-black uppercase tracking-widest text-[11px] flex items-center justify-center gap-3 hover:bg-slate-800 transition-all disabled:opacity-20 disabled:grayscale disabled:cursor-not-allowed shadow-xl"
            >
              <ShieldCheck size={18} /> I Acknowledge & Proceed
            </button>
            <button 
              onClick={onCancel}
              className="py-4 px-8 bg-white border border-slate-200 text-slate-500 rounded-2xl font-black uppercase tracking-widest text-[11px] hover:bg-slate-50 transition-all"
            >
              Cancel
            </button>
          </div>
        </div>
        
        <div className="h-2 bg-gradient-to-r from-blue-600 via-indigo-600 to-blue-600" />
      </motion.div>
    </div>
  );
};

export default PatientDataNotice;
