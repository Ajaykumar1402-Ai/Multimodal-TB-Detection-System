import React, { useState, useEffect } from 'react';

const BACKEND_URL = window.location.hostname === 'localhost'
  ? 'http://localhost:8000'
  : 'https://multimodal-tb-detection-system.onrender.com';

export const ServerStatus = () => {
  const [status, setStatus] = useState('checking'); 
  // 'checking' | 'awake' | 'sleeping' | 'waking'

  useEffect(() => {
    checkServerStatus();
  }, []);

  const checkServerStatus = async () => {
    setStatus('checking');
    try {
      const res = await fetch(`${BACKEND_URL}/health`, {
        signal: AbortSignal.timeout(5000)
      });
      if (res.ok) {
        setStatus('awake');
      } else {
        setStatus('sleeping');
      }
    } catch {
      setStatus('sleeping');
    }
  };

  const wakeServer = async () => {
    setStatus('waking');
    const startTime = Date.now();
    
    // Poll health endpoint until server responds or 60 seconds elapses
    while (Date.now() - startTime < 60000) {
      try {
        const res = await fetch(`${BACKEND_URL}/health`, {
          signal: AbortSignal.timeout(8000)
        });
        if (res.ok) {
          setStatus('awake');
          return;
        }
      } catch { }
      await new Promise(r => setTimeout(r, 3000));
    }
    setStatus('sleeping'); // Failed to wake after 60 seconds
  };

  const statusConfig = {
    checking: { 
      borderColor: 'border-slate-300 dark:border-slate-700',
      bgColor: 'bg-slate-50 dark:bg-slate-900/40',
      textColor: 'text-slate-500 dark:text-slate-400',
      icon: '⏳', 
      text: 'Synchronizing with Medical Engine...',
      showButton: false 
    },
    awake: { 
      borderColor: 'border-emerald-500/30 dark:border-emerald-500/20',
      bgColor: 'bg-emerald-500/5 dark:bg-emerald-500/10 backdrop-blur-md',
      textColor: 'text-emerald-600 dark:text-emerald-400',
      icon: '🟢', 
      text: 'AI Diagnostic Engine Online — Ready',
      showButton: false 
    },
    sleeping: { 
      borderColor: 'border-amber-500/30 dark:border-amber-500/20',
      bgColor: 'bg-amber-500/5 dark:bg-amber-500/10 backdrop-blur-md',
      textColor: 'text-amber-600 dark:text-amber-400',
      icon: '🟡', 
      text: 'AI Diagnostic Engine Sleeping — Activation Recommended',
      showButton: true 
    },
    waking: { 
      borderColor: 'border-blue-500/30 dark:border-blue-500/20',
      bgColor: 'bg-blue-500/5 dark:bg-blue-500/10 backdrop-blur-md',
      textColor: 'text-blue-600 dark:text-blue-400',
      icon: '⏳', 
      text: 'Activating AI Diagnostic Engine... (20-30 seconds)',
      showButton: false 
    },
  };

  const cfg = statusConfig[status];

  return (
    <div className={`p-4 border rounded-xl flex flex-col md:flex-row items-center gap-4 mb-6 transition-all duration-300 ${cfg.bgColor} ${cfg.borderColor}`}>
      <div className="flex items-center gap-3 flex-1 w-full">
        <span className="text-xl shrink-0 animate-pulse">{cfg.icon}</span>
        <span className={`font-semibold text-sm md:text-base flex-1 ${cfg.textColor}`}>
          {cfg.text}
        </span>
      </div>
      
      {cfg.showButton && (
        <button
          onClick={wakeServer}
          className="w-full md:w-auto px-5 py-2 bg-amber-500 hover:bg-amber-600 active:scale-95 text-white font-semibold text-sm rounded-lg cursor-pointer transition-all shadow-md shadow-amber-500/10"
        >
          Wake Up Server
        </button>
      )}
      
      {status === 'waking' && (
        <div className="w-full md:w-48 h-2.5 bg-slate-200 dark:bg-slate-800 rounded-full overflow-hidden shrink-0 mt-2 md:mt-0">
          <div 
            className="h-full bg-blue-500 rounded-full"
            style={{
              animation: 'wakeProgress 30s linear forwards'
            }}
          />
        </div>
      )}

      {/* Embedded wake progress animation styles */}
      <style>{`
        @keyframes wakeProgress {
          0% { width: 0%; }
          100% { width: 100%; }
        }
      `}</style>
    </div>
  );
};
