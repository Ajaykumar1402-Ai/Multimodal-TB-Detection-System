import React from 'react';
import { motion } from 'framer-motion';

export const GlassCard = ({ children, className = "", title, icon: Icon, badge }) => (
  <motion.div 
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    className={`glass-panel p-6 relative overflow-hidden group ${className}`}
  >
    {title && (
      <div className="flex items-center justify-between mb-6 relative z-10">
        <div className="flex items-center gap-3">
           {Icon && <div className="p-2 bg-blue-600/10 rounded-lg text-blue-600 group-hover:bg-blue-600 group-hover:text-white transition-all duration-500"><Icon size={18} /></div>}
           <h3 className="text-xs font-black text-slate-800 uppercase tracking-widest">{title}</h3>
        </div>
        {badge}
      </div>
    )}
    <div className="relative z-10">{children}</div>
    <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/5 blur-[60px] rounded-full group-hover:bg-blue-500/10 transition-all duration-700" />
  </motion.div>
);

export default GlassCard;
