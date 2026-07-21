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

    patients = relationship("Patient", back_populates="doctor")

class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    age = Column(Integer)
    gender = Column(String)
    doctor_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    doctor = relationship("User", back_populates="patients")
    records = relationship("DiagnosisRecord", back_populates="patient")

class DiagnosisRecord(Base):
    __tablename__ = "diagnosis_records"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    
    # Clinical inputs
    cough_duration_weeks = Column(Integer, default=0)
    fever = Column(Integer, default=0) # boolean 0/1
    weight_loss = Column(Integer, default=0) # 0/1
    night_sweats = Column(Integer, default=0) # 0/1
    
    # Image reference
    xray_image_path = Column(String, nullable=True)

    # ML Results
    cnn_probability = Column(Float, default=0.0)
    clinical_probability = Column(Float, default=0.0)
    final_tb_probability = Column(Float, default=0.0)
    risk_level = Column(String) # Low, Medium, High
    recommendations = Column(String)
    
    date = Column(DateTime, default=datetime.datetime.utcnow)

    patient = relationship("Patient", back_populates="records")
