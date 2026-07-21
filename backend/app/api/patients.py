from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..db import database, models
from .auth_utils import verify_password
from pydantic import BaseModel

router = APIRouter()

class PatientCreate(BaseModel):
    name: str
    age: int
    gender: str
    doctor_id: int

@router.post("/")
def create_patient(patient: PatientCreate, db: Session = Depends(database.get_db)):
    new_patient = models.Patient(**patient.dict())
    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)
    return new_patient

@router.get("/{doctor_id}")
def get_patients_for_doctor(doctor_id: int, db: Session = Depends(database.get_db)):
    patients = db.query(models.Patient).filter(models.Patient.doctor_id == doctor_id).all()
    return patients

@router.get("/history/{patient_id}")
def get_patient_history(patient_id: int, db: Session = Depends(database.get_db)):
    records = db.query(models.DiagnosisRecord).filter(models.DiagnosisRecord.patient_id == patient_id).order_by(models.DiagnosisRecord.date.desc()).all()
    return records
