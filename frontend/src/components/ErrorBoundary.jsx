import React from 'react';
import { AlertCircle, RefreshCcw } from 'lucide-react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("[CRITICAL SYSTEM ERROR]:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-slate-50 flex items-center justify-center p-6 font-sans">
          <div className="max-w-md w-full glass-panel p-10 text-center border-rose-100 shadow-[0_40px_100px_rgba(244,63,94,0.1)]">
            <div className="w-20 h-20 bg-rose-50 rounded-full flex items-center justify-center mx-auto mb-8 text-rose-500">
              <AlertCircle size={40} />
            </div>
            <h2 className="text-2xl font-black text-slate-900 tracking-tight mb-4">Diagnostic Pipeline Interrupted</h2>
            <p className="text-slate-500 text-sm leading-relaxed mb-10 font-medium">
              A synchronization error occurred in the neural synthesis layer. The clinical registry remains secure.
            </p>
            <button
              onClick={() => window.location.reload()}
              className="w-full flex items-center justify-center gap-3 bg-blue-600 text-white font-black uppercase tracking-widest text-xs py-4 rounded-2xl hover:bg-blue-700 transition-all shadow-xl shadow-blue-600/20"
            >
              <RefreshCcw size={18} /> Restore Connection
            </button>
            <p className="mt-8 text-[10px] font-black text-slate-300 uppercase tracking-widest">
              Error Ref: {this.state.error?.message?.slice(0, 40) || 'Unknown Core Exception'}
            </p>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
