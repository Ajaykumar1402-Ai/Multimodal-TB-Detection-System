import axios from 'axios';

const API_URL = window.location.hostname === 'localhost'
  ? 'http://localhost:8000/api'
  : 'https://multimodal-tb-detection-system.onrender.com/api';

const api = axios.create({
  baseURL: API_URL,
  timeout: 60000,
});

api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) config.headers.Authorization = `Bearer ${token}`;
    return config;
  },
  (error) => Promise.reject(error)
);

api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const { config, response } = error;

    const maxRetries = 4;
    config.retryCount = config.retryCount || 0;

    const isTimeout = error.code === 'ECONNABORTED';
    const isNetworkError = !response;
    const isServerWaking = response?.status === 503 || response?.status === 504;
    const isAuthError = response?.status === 401 || response?.status === 403;

    if (isAuthError) {
      localStorage.removeItem('token');
      localStorage.removeItem('username');
      if (window.location.pathname !== '/login') {
        window.location.href = '/login';
      }
      return Promise.reject(error);
    }

    if (
      config.retryCount < maxRetries &&
      (isTimeout || isServerWaking || (isNetworkError && window.location.hostname !== 'localhost'))
    ) {
      config.retryCount += 1;
      const delay = 3000 * Math.pow(2, config.retryCount - 1); // 3s, 6s, 12s, 24s
      console.warn(`[API] Server waking. Retry ${config.retryCount}/${maxRetries} in ${delay}ms...`);
      window.dispatchEvent(new CustomEvent('api-retry', { detail: { retryCount: config.retryCount, maxRetries } }));
      await new Promise((resolve) => setTimeout(resolve, delay));
      return api(config);
    }

    if (isTimeout || isServerWaking || isNetworkError) {
      error.message = `[INITIALIZING] Medical Engine Synchronizing (${config.retryCount}/${maxRetries}). Please wait a moment and try again.`;
    }

    return Promise.reject(error);
  }
);

export const login = async (email, password) => {
  const response = await api.post('/auth/login', { email, password });
  return response.data;
};

export const register = async (email, password, full_name) => {
  const response = await api.post('/auth/signup', { email, password, full_name });
  return response.data;
};

export const requestPasswordReset = async (email) => {
  const response = await api.post('/auth/request-reset', { email });
  return response.data;
};

export const uploadDiagnosis = async (formData, options = {}) => {
  const response = await api.post('/inference/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    ...options,
  });
  return response.data;
};

export const getPatients = async (doctorId) => {
  const response = await api.get(`/patients/${doctorId}`);
  return response.data;
};

export const createPatient = async (patientData) => {
  const response = await api.post('/patients/', patientData);
  return response.data;
};

export const getDashboardStats = async () => {
  const response = await api.get('/stats/dashboard');
  return response.data;
};

export const getAllDiagnoses = async (search = '') => {
  const response = await api.get(`/stats/all${search ? `?search=${encodeURIComponent(search)}` : ''}`);
  return response.data;
};

export const getPatientHistory = async (patientId) => {
  const response = await api.get(`/stats/patient/${patientId}`);
  return response.data;
};

export const logConsent = async (sessionId) => {
  const response = await api.post('/stats/log-consent', { session_id: sessionId });
  return response.data;
};

// BUG H-01: Standard XHR for real-time upload progress tracking
export const uploadWithProgress = (formData, onProgress) => {
  let xhr;
  const promise = new Promise((resolve, reject) => {
    xhr = new XMLHttpRequest();
    const token = localStorage.getItem('token');
    
    xhr.open('POST', `${API_URL}/inference/predict`, true);
    if (token) xhr.setRequestHeader('Authorization', `Bearer ${token}`);

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        const percentComplete = (event.loaded / event.total) * 100;
        onProgress(percentComplete, event.loaded, event.total);
      }
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch (e) {
          reject({ message: 'Invalid response from server' });
        }
      } else {
        try {
          const error = JSON.parse(xhr.responseText);
          error.status = xhr.status;
          reject(error);
        } catch (e) {
          reject({ message: xhr.statusText || 'Upload failed', status: xhr.status });
        }
      }
    };

    xhr.onerror = () => reject({ message: 'Network connection failure' });
    xhr.onabort = () => reject({ message: 'Upload cancelled by user', aborted: true });
    
    xhr.send(formData);
  });

  return {
    promise,
    cancel: () => xhr && xhr.abort()
  };
};

export default api;
