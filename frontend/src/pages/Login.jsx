import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { login, register } from '../services/api';
import { BrainCircuit, Loader2 } from 'lucide-react';
import { toast } from 'react-hot-toast';
import { motion } from 'framer-motion';

export default function Login() {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      if (isLogin) {
        const data = await login(email, password);
        localStorage.setItem('token', data.access_token);
        localStorage.setItem('username', data.user_name);
        toast.success(`Welcome back, ${data.user_name}`);
        navigate('/dashboard');
      } else {
        await register(email, password, fullName);
        toast.success('Registration successful. Please log in.');
        setIsLogin(true);
      }
    } catch (err) {
      toast.error(err.response?.data?.detail || 'An error occurred. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="glass-panel w-full max-w-md p-8 relative overflow-hidden"
      >
        <div className="absolute top-0 right-0 -mr-8 -mt-8 w-32 h-32 bg-primary/20 rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 left-0 -ml-8 -mb-8 w-32 h-32 bg-secondary/20 rounded-full blur-3xl"></div>
        
        <div className="relative z-10 flex flex-col items-center mb-8">
          <div className="p-3 bg-primary/10 rounded-2xl mb-4 text-primary">
            <BrainCircuit size={40} />
          </div>
          <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-secondary">
            {isLogin ? 'Sign in to TB-Vision' : 'Create an Account'}
          </h2>
          <p className="text-textmuted mt-2 text-sm text-center">
            AI-powered Multimodal Tuberculosis Detection
          </p>
        </div>

        <form onSubmit={handleSubmit} className="relative z-10 space-y-4">
          {!isLogin && (
            <div>
              <input
                type="text"
                placeholder="Full Name (Dr. Smith)"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                className="input-field"
                required
              />
            </div>
          )}
          <div>
            <input
              type="email"
              placeholder="Email Address"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="input-field"
              required
            />
          </div>
          <div>
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="input-field"
              required
            />
          </div>
          <button type="submit" className="btn-primary mt-6 flex justify-center items-center" disabled={loading}>
             {loading ? <Loader2 className="animate-spin" size={20} /> : (isLogin ? 'Log In' : 'Sign Up')}
          </button>
        </form>

        <div className="relative z-10 mt-6 text-center">
          <button
            onClick={() => setIsLogin(!isLogin)}
            className="text-primary hover:text-white transition-colors text-sm"
          >
            {isLogin ? "Don't have an account? Sign up" : 'Already have an account? Log in'}
          </button>
        </div>
      </motion.div>
    </div>
  );
}
