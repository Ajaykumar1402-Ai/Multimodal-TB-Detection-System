import React from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import axios from 'axios';
import { 
  ArrowRight, Star, Activity, Play, Plus, 
  BrainCircuit, ActivitySquare, Zap, ShieldCheck,
  UploadCloud, ListChecks, Stethoscope, FileText
} from 'lucide-react';

// Animation variants for scroll reveals
const fadeInUp = {
  hidden: { opacity: 0, y: 40 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: "easeOut" } }
};

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.15 }
  }
};

export default function Landing() {
  const navigate = useNavigate();

  // Pre-warm the backend (wake up Render free tier)
  React.useEffect(() => {
    const API_BASE = window.location.hostname === 'localhost' 
      ? 'http://localhost:8000' 
      : 'https://multimodal-tb-detection-system.onrender.com';
    
    // Silent ping to wake up the server
    axios.get(API_BASE).catch(() => {});
  }, []);

  return (
    <div className="relative min-h-screen bg-[#f8fafc] overflow-hidden font-sans text-slate-900">
      
      {/* Global Background Glows */}
      <div className="absolute top-0 left-0 w-full h-[800px] bg-gradient-to-b from-[#e0f2fe] to-transparent opacity-50 pointer-events-none z-0" />
      <div className="absolute top-[-10%] left-[-5%] w-[45%] h-[55%] rounded-full bg-gradient-to-r from-[#3B82F6] to-[#06B6D4] blur-[150px] opacity-[0.12] pointer-events-none z-0" />

      {/* --- Sticky Navbar --- */}
      <nav className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 lg:px-12 py-4 backdrop-blur-[30px] border-b border-black/5"
           style={{ backgroundColor: 'rgba(255, 255, 255, 0.6)' }}>
        <div className="flex items-center gap-2 cursor-pointer" onClick={() => navigate('/')}>
           <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center text-white shadow-sm">
              <Activity size={18} strokeWidth={2.5} />
           </div>
           <div className="font-[Poppins] font-bold text-xl tracking-tight text-slate-800">TB-Vision Pro</div>
        </div>
        
        <div className="hidden md:flex items-center gap-8 font-medium text-slate-600 text-[14px]">
          <a href="#home" className="hover:text-blue-600 transition-colors">Home</a>
          <a href="#features" className="hover:text-blue-600 transition-colors">Features</a>
          <a href="#how-it-works" className="hover:text-blue-600 transition-colors">How it Works</a>
          <span onClick={() => navigate('/dashboard')} className="hover:text-blue-600 cursor-pointer transition-colors">Dashboard</span>
        </div>

        <button 
          onClick={() => navigate('/login')}
          className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 transition-all font-semibold px-5 py-2 rounded-xl text-[14px] text-white shadow-sm"
        >
          Login / Get Started <ArrowRight size={16} />
        </button>
      </nav>

      {/* --- Main Content Wrapper (1600px Max) --- */}
      <div className="relative z-10 w-full max-w-[1600px] mx-auto px-6 lg:px-12 flex flex-col pt-[100px] -webkit-font-smoothing-antialiased">
        
        {/* --- 1. HERO SECTION --- */}
        <section id="home" className="flex flex-col lg:flex-row items-center justify-between min-h-[85vh] gap-16 lg:gap-8 pt-8 lg:pt-0">
          
          {/* Left Text */}
          <motion.div initial="hidden" animate="visible" variants={staggerContainer} className="flex-1 max-w-[680px] z-20">
            <motion.div variants={fadeInUp} className="flex items-center gap-3 mb-6 bg-white/60 backdrop-blur-md px-4 py-2 rounded-full w-fit border border-blue-100 shadow-sm cursor-default">
              <div className="flex text-[#F59E0B]">
                <Star size={14} fill="currentColor" stroke="none" />
                <Star size={14} fill="currentColor" stroke="none" />
                <Star size={14} fill="currentColor" stroke="none" />
                <Star size={14} fill="currentColor" stroke="none" />
                <Star size={14} fill="currentColor" stroke="none" />
              </div>
              <span className="text-sm font-semibold text-slate-700 tracking-tight">Trusted by healthcare professionals • 96.7% accuracy</span>
            </motion.div>

            <motion.h1 variants={fadeInUp} className="font-[Poppins] font-bold text-[50px] md:text-[64px] leading-[1.05] tracking-[-1px] text-slate-900 mb-6 drop-shadow-sm">
              Detect Tuberculosis Faster with AI
            </motion.h1>
            
            <motion.p variants={fadeInUp} className="text-[18px] text-slate-600 font-medium tracking-[-0.5px] mb-10 leading-relaxed max-w-[580px]">
              Multimodal AI combining X-rays and clinical symptoms for accurate diagnosis in seconds. Designed for modern medical clarity.
            </motion.p>

            <motion.div variants={fadeInUp} className="flex flex-wrap items-center gap-4">
               <button 
                 onClick={() => navigate('/diagnosis')}
                 className="group flex items-center gap-3 font-semibold text-base text-white px-6 py-3 rounded-[16px] transition-all hover:scale-[1.03] hover:shadow-lg bg-gradient-to-r from-blue-600 to-blue-500"
               >
                 Start Diagnosis
                 <div className="w-8 h-8 rounded-full bg-white/20 flex items-center justify-center text-white transform group-hover:translate-x-1 transition-transform">
                   <Plus size={18} strokeWidth={2.5} />
                 </div>
               </button>

               <button 
                 onClick={() => document.getElementById('demo-section').scrollIntoView({ behavior: 'smooth' })}
                 className="group flex items-center gap-2 font-semibold text-base text-slate-700 px-6 py-3 rounded-[16px] transition-all hover:bg-slate-100 border border-slate-200"
               >
                 <Play size={18} className="text-blue-500 fill-current" /> View Demo
               </button>
            </motion.div>
          </motion.div>

          {/* Right Visual (Abstract 3D Lung Scanning Effect simulation via Video/Orb) */}
          <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 1 }} className="flex-1 relative w-full flex justify-center items-center h-[500px] lg:h-[700px] pointer-events-none z-10">
            {/* Visual representation of 3D lung highlight scanning */}
            <div className="absolute w-[400px] h-[400px] bg-gradient-to-tr from-[#EF4444] to-[#F59E0B] rounded-full blur-[100px] opacity-20 animate-pulse"></div>
            <video 
              autoPlay loop muted playsInline
              className="absolute w-full h-full object-contain scale-125 translate-x-0 lg:translate-x-10"
              style={{ mixBlendMode: 'screen', filter: 'hue-rotate(-20deg) saturate(140%) brightness(1.1)' }}
            >
              <source src="https://future.co/images/homepage/glassy-orb/orb-purple.webm" type="video/webm" />
            </video>
          </motion.div>
        </section>

        {/* --- 2. TRUST SECTION --- */}
        <motion.section 
          initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-100px" }} variants={fadeInUp}
          className="w-full flex flex-col justify-center items-center py-16 mt-8 border-y border-slate-200/50"
        >
           <p className="text-slate-400 text-sm font-semibold uppercase tracking-[2px] mb-10">Trusted by modern healthcare systems</p>
           <div className="flex flex-wrap justify-center items-center gap-[60px] md:gap-[100px] opacity-50 grayscale hover:grayscale-0 transition-all duration-700">
              <div className="flex flex-col items-center gap-1 cursor-default"><Activity size={28}/><span className="font-bold text-lg">NationalHealth</span></div>
              <div className="flex flex-col items-center gap-1 cursor-default"><BrainCircuit size={28}/><span className="font-bold text-lg">AI Labs</span></div>
              <div className="flex flex-col items-center gap-1 cursor-default"><Stethoscope size={28}/><span className="font-bold text-lg">MedCorp</span></div>
              <div className="flex flex-col items-center gap-1 cursor-default">
                 <span className="font-black text-2xl text-blue-600">{`< 15s`}</span><span className="font-bold text-xs uppercase tracking-widest text-slate-800">Speed</span>
              </div>
           </div>
        </motion.section>

        {/* --- 3. FEATURES SECTION --- */}
        <motion.section id="features" initial="hidden" whileInView="visible" viewport={{ once: true }} variants={staggerContainer} className="pt-32 pb-20">
          <motion.div variants={fadeInUp} className="text-center mb-16">
             <h2 className="font-[Poppins] font-bold text-4xl text-slate-900 mb-4">Intelligent Medical Clarity</h2>
             <p className="text-slate-500 font-medium max-w-2xl mx-auto">Leveraging deep learning and clinical parameters to provide an explainable, highly accurate pipeline for TB screening.</p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
             {/* Card 1 */}
             <motion.div variants={fadeInUp} className="bg-white p-8 rounded-[20px] shadow-[0_4px_20px_rgba(0,0,0,0.03)] border border-slate-100 hover:-translate-y-2 transition-transform duration-300">
                <div className="w-12 h-12 bg-blue-50 text-blue-600 rounded-2xl flex items-center justify-center mb-6"><FileText size={24} /></div>
                <h3 className="font-bold text-lg text-slate-800 mb-3">AI X-ray Analysis</h3>
                <p className="text-sm text-slate-500 leading-relaxed">Advanced CNNs rapidly analyze chest radiographs, identifying subtle infiltrates associated with Tuberculosis.</p>
             </motion.div>
             {/* Card 2 */}
             <motion.div variants={fadeInUp} className="bg-white p-8 rounded-[20px] shadow-[0_4px_20px_rgba(0,0,0,0.03)] border border-slate-100 hover:-translate-y-2 transition-transform duration-300">
                <div className="w-12 h-12 bg-cyan-50 text-cyan-600 rounded-2xl flex items-center justify-center mb-6"><ActivitySquare size={24} /></div>
                <h3 className="font-bold text-lg text-slate-800 mb-3">Symptom Prediction</h3>
                <p className="text-sm text-slate-500 leading-relaxed">Patient history and clinical symptoms are fused with imaging data to drastically decrease false positives.</p>
             </motion.div>
             {/* Card 3 */}
             <motion.div variants={fadeInUp} className="bg-white p-8 rounded-[20px] shadow-[0_4px_20px_rgba(0,0,0,0.03)] border border-slate-100 hover:-translate-y-2 transition-transform duration-300">
                <div className="w-12 h-12 bg-emerald-50 text-emerald-600 rounded-2xl flex items-center justify-center mb-6"><Zap size={24} /></div>
                <h3 className="font-bold text-lg text-slate-800 mb-3">Real-Time Results</h3>
                <p className="text-sm text-slate-500 leading-relaxed">Inference runs natively fast, providing high-probability differential results in under 15 seconds.</p>
             </motion.div>
             {/* Card 4 */}
             <motion.div variants={fadeInUp} className="bg-white p-8 rounded-[20px] shadow-[0_4px_20px_rgba(0,0,0,0.03)] border border-slate-100 hover:-translate-y-2 transition-transform duration-300">
                <div className="w-12 h-12 bg-purple-50 text-purple-600 rounded-2xl flex items-center justify-center mb-6"><ShieldCheck size={24} /></div>
                <h3 className="font-bold text-lg text-slate-800 mb-3">Explainable AI</h3>
                <p className="text-sm text-slate-500 leading-relaxed">Deep learning is no longer a black box. Heatmaps clearly highlight the precise lung areas driving the model's prediction.</p>
             </motion.div>
          </div>
        </motion.section>

        {/* --- 4. HOW IT WORKS SECTION --- */}
        <motion.section id="how-it-works" initial="hidden" whileInView="visible" viewport={{ once: true }} variants={staggerContainer} className="pt-20 pb-32">
           <motion.div variants={fadeInUp} className="text-center mb-16">
             <h2 className="font-[Poppins] font-bold text-4xl text-slate-900 mb-4">Pipeline Workflow</h2>
             <p className="text-slate-500 font-medium max-w-2xl mx-auto">Seamless diagnostic protocol integrated directly into standard clinical workflows.</p>
          </motion.div>

          <div className="flex flex-col md:flex-row justify-between items-start relative max-w-5xl mx-auto px-4">
             {/* Connecting Line Desktop */}
             <div className="hidden md:block absolute top-[40px] left-[10%] right-[10%] h-[2px] bg-slate-200 z-0"></div>

             {[
               { icon: <UploadCloud size={28}/>, title: "1. Upload X-ray", desc: "Drag & drop standard chest radiography standards." },
               { icon: <ListChecks size={28}/>, title: "2. Input Symptoms", desc: "Log clinical qualifiers like fever, weight loss, or hemoptysis." },
               { icon: <BrainCircuit size={28}/>, title: "3. Neural Scan", desc: "State-of-the-art multimodal fusion network evaluates input." },
               { icon: <FileText size={28}/>, title: "4. Digital Report", desc: "Instant clinical PDF generation with risk stratification." }
             ].map((step, idx) => (
                <motion.div key={idx} variants={fadeInUp} className="relative z-10 flex flex-col items-center text-center max-w-[200px] mx-auto md:mx-0 mb-10 md:mb-0">
                   <div className="w-20 h-20 bg-white rounded-full border-[4px] border-[#FAFAFA] shadow-[0_10px_30px_rgba(0,0,0,0.08)] flex items-center justify-center text-blue-600 mb-6">
                      {step.icon}
                   </div>
                   <h4 className="font-bold text-slate-800 mb-2">{step.title}</h4>
                   <p className="text-xs text-slate-500 leading-relaxed">{step.desc}</p>
                </motion.div>
             ))}
          </div>
        </motion.section>

        {/* --- 5. LIVE DEMO PREVIEW --- */}
        <motion.section id="demo-section" initial={{ opacity: 0, y: 50 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.8 }} className="py-20 flex justify-center">
           <div className="w-full max-w-5xl relative rounded-[40px] p-2 bg-white/40 backdrop-blur-xl shadow-[0_40px_100px_rgba(0,0,0,0.1)] border border-white/50 overflow-hidden transform perspective-1000 rotate-x-1">
              <div 
                 className="w-full h-[600px] rounded-[32px] bg-slate-900 overflow-hidden relative group cursor-pointer"
                 onClick={() => {
                   const v = document.getElementById('promo-video');
                   if (v.paused) v.play(); else v.pause();
                 }}
              >
                 {/* Premium Demo Video */}
                 <video 
                   id="promo-video"
                   className="w-full h-full rounded-[32px] border-none opacity-90 group-hover:opacity-100 transition-opacity duration-700 object-cover"
                   src="/demo_video.mp4"
                   autoPlay 
                   muted 
                   loop 
                   playsInline
                 ></video>

                 {/* Information Overlay */}
                 <div className="absolute inset-x-0 bottom-0 p-12 bg-gradient-to-t from-slate-950 via-slate-950/40 to-transparent pointer-events-none">
                    <div className="flex items-center gap-3 mb-4">
                       <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
                       <span className="text-[10px] font-black text-blue-400 uppercase tracking-[0.3em]">Live Platform Demonstration</span>
                    </div>
                    <h3 className="text-3xl font-black text-white tracking-tight mb-3">Clinical Intelligence in Action.</h3>
                    <p className="text-slate-400 text-sm max-w-xl leading-relaxed">
                       Observe the real-time fusion of radiology and clinical parameters. Our platform stradifies risk and provides explainable heatmaps in under 15 seconds.
                    </p>
                 </div>

                 {/* Play Button Overlay (Visible on Hover) */}
                 <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-all duration-500 bg-slate-950/20 backdrop-blur-[2px]">
                    <div className="w-24 h-24 bg-white/10 backdrop-blur-3xl rounded-full border border-white/20 flex items-center justify-center pl-2 shadow-2xl scale-90 group-hover:scale-100 transition-transform">
                       <Play size={40} className="text-white fill-current" />
                    </div>
                 </div>
              </div>
           </div>
        </motion.section>

        {/* --- 6. BIG CALL TO ACTION --- */}
        <motion.section initial="hidden" whileInView="visible" viewport={{ once: true }} variants={fadeInUp} className="py-32">
           <div className="bg-gradient-to-br from-[#0F172A] to-[#1E293B] rounded-[30px] p-12 md:p-20 text-center relative overflow-hidden shadow-2xl">
              {/* Decor */}
              <div className="absolute top-0 right-0 w-64 h-64 bg-blue-500 rounded-full blur-[100px] opacity-20"></div>
              <div className="absolute bottom-0 left-0 w-64 h-64 bg-cyan-500 rounded-full blur-[100px] opacity-20"></div>
              
              <div className="relative z-10 w-full max-w-2xl mx-auto flex flex-col items-center text-center">
                 <h2 className="font-[Poppins] font-bold text-4xl md:text-5xl text-white mb-6">Start detecting TB smarter today.</h2>
                 <p className="text-slate-300 text-lg mb-10 max-w-lg">Implement the next generation of respiratory AI into your clinical workflow with zero onboarding friction.</p>
                 <button 
                   onClick={() => navigate('/login')}
                   className="text-lg font-bold text-white bg-gradient-to-r from-blue-600 to-blue-500 px-10 py-5 rounded-full shadow-[0_10px_30px_rgba(37,99,235,0.4)] hover:shadow-[0_10px_40px_rgba(37,99,235,0.6)] hover:scale-105 transition-all w-fit"
                 >
                   Try TB-Vision Pro Now
                 </button>
              </div>
           </div>
        </motion.section>
      </div>

      {/* --- 7. FOOTER --- */}
      <footer className="w-full border-t border-slate-200 bg-white py-12 relative z-10 text-slate-500 text-sm">
         <div className="max-w-[1600px] mx-auto px-6 lg:px-12 flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center gap-2 mb-4 md:mb-0">
               <div className="w-6 h-6 rounded bg-slate-800 flex items-center justify-center text-white"><Activity size={12} /></div>
               <span className="font-bold text-slate-800">TB-Vision Pro © 2026</span>
            </div>
            <div className="flex gap-8 font-medium">
               <a href="#" className="hover:text-blue-600 transition-colors">About</a>
               <a href="#" className="hover:text-blue-600 transition-colors">Privacy</a>
               <a href="#" className="hover:text-blue-600 transition-colors">Contact</a>
            </div>
         </div>
      </footer>
    </div>
  );
}
