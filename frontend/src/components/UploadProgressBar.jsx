import React from 'react';
import { motion } from 'framer-motion';

const UploadProgressBar = ({ progress, total, loaded, onCancel }) => {
  const percentage = Math.round(progress);
  const totalMB = (total / (1024 * 1024)).toFixed(1);
  const loadedMB = (loaded / (1024 * 1024)).toFixed(1);

  return (
    <div className="w-full bg-slate-50 border border-slate-200 rounded-3xl p-6 shadow-sm overflow-hidden relative group">
      <div className="flex justify-between items-center mb-4 relative z-10">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-blue-600 animate-pulse" />
          <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">
            Transmitting Clinical Data...
          </p>
        </div>
        <span className="text-xs font-black text-blue-600 font-[Poppins]">
          {percentage}%
        </span>
      </div>

      <div className="h-3 w-full bg-slate-200 rounded-full mb-4 relative z-10 overflow-hidden">
        <motion.div 
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          className="h-full bg-gradient-to-r from-blue-600 to-cyan-500 rounded-full"
        />
      </div>

      <div className="flex justify-between items-center relative z-10">
        <p className="text-[10px] font-bold text-slate-500">
          {loadedMB} MB of {totalMB} MB uploaded
        </p>
        <button 
          onClick={onCancel}
          className="text-[10px] font-black text-rose-500 hover:text-rose-700 uppercase tracking-widest transition-all p-2 hover:bg-rose-50 rounded-lg"
        >
          Cancel Upload
        </button>
      </div>

      {/* Decorative background glow */}
      <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/5 blur-[50px] rounded-full pointer-events-none" />
    </div>
  );
};

export default UploadProgressBar;
