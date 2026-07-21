import axios from 'axios';

const API_URL = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_URL,
});

// Add a request interceptor to include the auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

export const login = async (email, password) => {
  const response = await api.post('/auth/login', { email, password });
  return response.data;
};

export const register = async (email, password, full_name) => {
  const response = await api.post('/auth/signup', { email, password, full_name });
  return response.data;
};

export const uploadDiagnosis = async (formData) => {
  const response = await api.post('/inference/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
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

export default api;
