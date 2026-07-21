import React, { useEffect, useState } from 'react';
import { getPatients } from '../services/api';
import { toast } from 'react-hot-toast';
import { Users, AlertTriangle, FileCheck, Activity } from 'lucide-react';
import { 
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer 
} from 'recharts';
import { motion } from 'framer-motion';

export default function Dashboard() {
  const [stats, setStats] = useState({ total: 0, highRisk: 0, resolved: 0 });
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(true);
  const username = localStorage.getItem('username') || 'Doctor';

  // Mock data fetching since we don't have a getDashboardStats endpoint
  useEffect(() => {
    // In a real app, we'd fetch actual patients and aggregate
    setTimeout(() => {
      setStats({
        total: 142,
        highRisk: 18,
        resolved: 124
      });
      
      setChartData([
        { name: 'Jan', screenings: 400, positives: 24 },
        { name: 'Feb', screenings: 300, positives: 18 },
        { name: 'Mar', screenings: 550, positives: 32 },
        { name: 'Apr', screenings: 450, positives: 25 },
        { name: 'May', screenings: 600, positives: 40 },
        { name: 'Jun', screenings: 800, positives: 55 },
      ]);
      setLoading(false);
    }, 1000);
  }, []);

  const StatCard = ({ title, value, icon: Icon, color, delay }) => (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      className="glass-panel p-6 flex items-center justify-between"
    >
      <div>
        <p className="text-textmuted text-sm uppercase tracking-wider font-semibold mb-1">{title}</p>
        <h3 className="text-3xl font-bold text-white">{value}</h3>
      </div>
      <div className={`p-4 rounded-xl bg-${color}-500/20 text-${color}-400`}>
        <Icon size={28} />
      </div>
    </motion.div>
  );

  if (loading) return <div className="h-full flex items-center justify-center"><div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div></div>;

  return (
    <div className="space-y-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Welcome Back, {username}</h1>
        <p className="text-textmuted">Here's your TB screening overview for today.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <StatCard title="Total Screenings" value={stats.total} icon={Users} color="blue" delay={0.1} />
        <StatCard title="High Risk Detected" value={stats.highRisk} icon={AlertTriangle} color="red" delay={0.2} />
        <StatCard title="Cleared Patients" value={stats.resolved} icon={FileCheck} color="green" delay={0.3} />
      </div>

      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, delay: 0.4 }}
        className="glass-panel p-6 h-[400px] mt-8"
      >
        <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
          <Activity className="text-primary" /> Screening Trends
        </h3>
        <ResponsiveContainer width="100%" height="85%">
          <LineChart data={chartData}>
            <XAxis dataKey="name" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" />
            <Tooltip 
              contentStyle={{ backgroundColor: '#121b2d', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
              itemStyle={{ color: '#fff' }}
            />
            <Line type="monotone" dataKey="screenings" stroke="#38bdf8" strokeWidth={3} dot={{ r: 4, fill: '#38bdf8' }} activeDot={{ r: 6 }} />
            <Line type="monotone" dataKey="positives" stroke="#ef4444" strokeWidth={3} dot={{ r: 4, fill: '#ef4444' }} />
          </LineChart>
        </ResponsiveContainer>
      </motion.div>
      
      {/* Model Performance Demo Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.5 }}
        className="glass-panel p-6 mt-8"
      >
        <h3 className="text-xl font-bold text-white mb-4">AI Model Performance</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div className="bg-white/5 p-4 rounded-xl border border-white/5">
                <p className="text-textmuted text-sm mb-1">Accuracy</p>
                <p className="text-2xl font-bold text-green-400">96.4%</p>
            </div>
            <div className="bg-white/5 p-4 rounded-xl border border-white/5">
                <p className="text-textmuted text-sm mb-1">Precision</p>
                <p className="text-2xl font-bold text-blue-400">94.2%</p>
            </div>
            <div className="bg-white/5 p-4 rounded-xl border border-white/5">
                <p className="text-textmuted text-sm mb-1">Recall</p>
                <p className="text-2xl font-bold text-purple-400">98.1%</p>
            </div>
            <div className="bg-white/5 p-4 rounded-xl border border-white/5">
                <p className="text-textmuted text-sm mb-1">F1 Score</p>
                <p className="text-2xl font-bold text-yellow-400">96.1%</p>
            </div>
        </div>
      </motion.div>

    </div>
  );
}
