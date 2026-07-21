from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship
import datetime
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    
    # Audit trail
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    patients = relationship("Patient", back_populates="doctor")

class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    age = Column(Integer)
    date_of_birth = Column(String, nullable=True)
    gender = Column(String)
    doctor_id = Column(Integer, ForeignKey("users.id"), index=True)
    
    # Audit trail
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    doctor = relationship("User", back_populates="patients")
    records = relationship("DiagnosisRecord", back_populates="patient")

class DiagnosisRecord(Base):
    __tablename__ = "diagnosis_records"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), index=True)
    
    # Clinical inputs
    visit_date = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    cough_duration_weeks = Column(Integer, default=0)
    fever = Column(Integer, default=0) # boolean 0/1
    weight_loss = Column(Integer, default=0) # 0/1
    night_sweats = Column(Integer, default=0) # 0/1
    no_symptoms = Column(Integer, default=0) # 0/1
    
    # Lab data
    sputum_test = Column(Integer, default=0) 
    genexpert_test = Column(Integer, default=0) 
    
    # AI & Explainability Metadata
    model_version = Column(String, default="v2.4.0-who-calibrated")
    inference_latency_ms = Column(Float, default=0.0)
    confidence_interval = Column(String, nullable=True) # e.g. "92-96%"
    grad_cam_heatmap = Column(String, nullable=True) # Path to heatmap
    report_path = Column(String, nullable=True) # PDF Report link
    
    # Image reference
    xray_image_path = Column(String, nullable=True)

    # ML Results
    cnn_probability = Column(Float, default=0.0)
    clinical_probability = Column(Float, default=0.0)
    final_tb_probability = Column(Float, default=0.0)
    risk_level = Column(String) # Low, Medium, High
    recommendations = Column(String)
    
    # Communication Tracking
    is_email_notified = Column(Integer, default=0) # 0/1
    notified_at = Column(DateTime, nullable=True)
    
    # Audit trail
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    patient = relationship("Patient", back_populates="records")

class ModelLog(Base):
    __tablename__ = "model_logs"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String, unique=True, index=True)
    architecture = Column(String)
    accuracy = Column(Float)
    f1_score = Column(Float)
    training_dataset = Column(String)
    last_calibrated = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Metadata for enterprise audit
    weights_hash = Column(String)
    is_active = Column(Integer, default=1)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class EmailLog(Base):
    __tablename__ = "email_logs"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String)
    error_message = Column(String, nullable=True)

class ConsentLog(Base):
    __tablename__ = "consent_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    session_id = Column(String, index=True)
    event_type = Column(String, default="ACKNOWLEDGEMENT")
    ip_obfuscated = Column(String, nullable=True) # For audit without PII
