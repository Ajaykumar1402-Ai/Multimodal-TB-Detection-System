import React, { createContext, useContext, useReducer } from 'react';

const DiagnosisContext = createContext();

const initialState = {
  result: null,
  file: null,
  preview: null,
  loading: false,
  uploadError: null,
  hasConsented: sessionStorage.getItem('tb_consent_ack') === 'true',
  formData: {
    patient_id: '',
    patient_name: '',
    age: 45,
    doctor_email: 'doctor@hospital.org',
    cough_duration_weeks: 0,
    fever: 0,
    weight_loss: 0,
    night_sweats: 0,
    sputum_test: 0,
    genexpert_test: 0,
    no_symptoms: 0
  }
};

function diagnosisReducer(state, action) {
  switch (action.type) {
    case 'SET_RESULT':
      return { ...state, result: action.payload, loading: false, uploadError: null };
    case 'SET_LOADING':
      return { ...state, loading: action.payload };
    case 'SET_ERROR':
      return { ...state, uploadError: action.payload, loading: false };
    case 'SET_FILE':
      return { ...state, file: action.payload.file, preview: action.payload.preview };
    case 'UPDATE_FORM':
      return { ...state, formData: { ...state.formData, ...action.payload } };
    case 'SET_CONSENT':
      return { ...state, hasConsented: action.payload };
    case 'RESET_ANALYSIS':
      // Revoke preview URL to free memory, but DO NOT clear sessionStorage
      // (clearing it would wipe the consent acknowledgement and break the app)
      if (state.preview) {
        URL.revokeObjectURL(state.preview);
      }
      return { ...initialState, hasConsented: state.hasConsented };
    default:
      return state;
  }
}

export const DiagnosisProvider = ({ children }) => {
  const [state, dispatch] = useReducer(diagnosisReducer, initialState);
  return (
    <DiagnosisContext.Provider value={{ state, dispatch }}>
      {children}
    </DiagnosisContext.Provider>
  );
};

export const useDiagnosis = () => {
  const context = useContext(DiagnosisContext);
  if (!context) throw new Error('useDiagnosis must be used within a DiagnosisProvider');
  return context;
};
