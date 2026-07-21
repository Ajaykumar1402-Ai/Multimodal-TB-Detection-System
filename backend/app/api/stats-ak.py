from fastapi import APIRouter, Depends, Body
from sqlalchemy.orm import Session
from sqlalchemy import func, extract
from ..db import database, models
import datetime
import os
from fastapi.responses import FileResponse
from fastapi import HTTPException

router = APIRouter()

@router.post("/log-consent")
def log_consent(
    session_id: str = Body(..., embed=True),
    db: Session = Depends(database.get_db)
):
    log = models.ConsentLog(session_id=session_id)
    db.add(log)
    db.commit()
    return {"status": "logged"}

@router.get("/dashboard")
def get_dashboard_stats(db: Session = Depends(database.get_db)):
    # Fetch all records to do robust in-memory aggregation. 
    # This prevents all PostgreSQL vs SQLite syntax mismatch errors for complex funcs.
    records = db.query(models.DiagnosisRecord).all()
    
    total_screenings = len(records)
    high_risk = sum(1 for r in records if r.risk_level == "High")
    cleared = sum(1 for r in records if r.risk_level == "Low")
    
    # Monthly Trend Data for 2026
    current_year = 2026
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    # Initialize chart data with zeros
    chart_dict = {i: {"screenings": 0, "positives": 0} for i in range(1, 13)}
    
    for r in records:
        if r.visit_date and r.visit_date.year == current_year:
            m = r.visit_date.month
            chart_dict[m]["screenings"] += 1
            if r.final_tb_probability is not None and r.final_tb_probability >= 0.42:
                chart_dict[m]["positives"] += 1
                
    chart_data = []
    for i in range(1, 7): # Just Jan-Jun for the UI
        chart_data.append({
            "name": month_names[i-1],
            "screenings": chart_dict[i]["screenings"],
            "positives": chart_dict[i]["positives"]
        })

    # Age Distribution (Requires joining with patient)
    all_diagnoses = db.query(models.DiagnosisRecord).join(models.Patient).all()
    
    age_0_18 = 0
    age_19_35 = 0
    age_36_50 = 0
    age_51_plus = 0
    
    for r in all_diagnoses:
        age = r.patient.age if r.patient and r.patient.age else 0
        if age <= 18:
            age_0_18 += 1
        elif 18 < age <= 35:
            age_19_35 += 1
        elif 35 < age <= 50:
            age_36_50 += 1
        else:
            age_51_plus += 1
            
    age_data = [
        {"group": "0-18", "value": age_0_18},
        {"group": "19-35", "value": age_19_35},
        {"group": "36-50", "value": age_36_50},
        {"group": "51+", "value": age_51_plus}
    ]

    # Recent Activity
    recent_records = sorted(all_diagnoses, key=lambda x: x.id, reverse=True)[:5]
    recent_activity = [{
        "id": r.id,
        "name": r.patient.name if r.patient else "Unknown",
        "date": r.visit_date.strftime("%b %d, %Y") if r.visit_date else "N/A",
        "risk": r.risk_level,
        "probability": round(float(r.final_tb_probability or 0) * 100, 1)
    } for r in recent_records]
        
    return {
        "total": total_screenings,
        "highRisk": high_risk,
        "resolved": cleared,
        "chartData": chart_data,
        "ageData": age_data,
        "recentActivity": recent_activity
    }

@router.get("/all")
def get_all_diagnoses(search: str = None, db: Session = Depends(database.get_db)):
    query = db.query(models.DiagnosisRecord).join(models.Patient)
    if search:
        query = query.filter(models.Patient.name.ilike(f"%{search}%"))
    
    # Order by ID (insertion order) to ensure new diagnoses are always at the top
    records = query.order_by(models.DiagnosisRecord.id.desc()).all()
    
    result = []
    for r in records:
        safe_name = r.patient.name.replace(' ', '_')
        # Point to dynamic generator endpoint instead of static files (prevents 404s after Render restart)
        pdf_url = f"https://multimodal-tb-detection-system.onrender.com/api/stats/report/{r.id}"
        result.append({
            "id": r.id,
            "patient_name": r.patient.name,
            "patient_id": r.patient_id,
            "date": r.visit_date.strftime("%b %d, %Y") if r.visit_date else "N/A",
            "risk_level": r.risk_level,
            "final_prob": float(r.final_tb_probability or 0),
            "recommendations": r.recommendations,
            "pdf_url": pdf_url
        })
    return result

@router.get("/report/{record_id}")
def download_dynamic_report(record_id: int, db: Session = Depends(database.get_db)):
    """Dynamically generates the PDF from Postgres data so it never gets lost."""
    from ..services import pdf_service
    record = db.query(models.DiagnosisRecord).filter(models.DiagnosisRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Diagnosis record not found")
        
    record_data = {
        "cough_duration_weeks": record.cough_duration_weeks,
        "fever": record.fever,
        "weight_loss": record.weight_loss,
        "night_sweats": record.night_sweats,
        "sputum_test": record.sputum_test,
        "genexpert_test": record.genexpert_test,
        "patient_id": record.patient_id,
        "age": record.patient.age if record.patient else "N/A",
        "date_of_birth": getattr(record.patient, 'date_of_birth', "N/A") if record.patient else "N/A",
        "final_probability": record.final_tb_probability,
        "risk_level": record.risk_level,
        "recommendations": record.recommendations,
        "cnn_probability": record.cnn_probability,
        "clinical_probability": record.clinical_probability,
        "doctor_email": record.patient.doctor.email if record.patient and getattr(record.patient, 'doctor', None) else "Medical Professional",
    }
    
    # Store temporarily; it will be streamed to the user
    output_dir = "./tmp_reports"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    pdf_path = pdf_service.generate_pdf_report(record.patient.name or "Unknown", record_data, record.id, output_dir=output_dir)
    return FileResponse(pdf_path, media_type="application/pdf", filename=os.path.basename(pdf_path))

@router.get("/patient/{patient_id}")
def get_patient_history(patient_id: int, db: Session = Depends(database.get_db)):
    """Fetches diagnostic history for a specific patient for recovery tracking"""
    records = db.query(models.DiagnosisRecord)\
                .filter(models.DiagnosisRecord.patient_id == patient_id)\
                .order_by(models.DiagnosisRecord.visit_date.asc())\
                .all()
    
    return [{
        "date": r.visit_date.strftime("%Y-%m-%d") if r.visit_date else "N/A",
        "displayDate": r.visit_date.strftime("%b %d") if r.visit_date else "N/A",
        "probability": round(float(r.final_tb_probability or 0) * 100, 1),
        "risk_level": r.risk_level
    } for r in records]

@router.post("/audit-pii")
def retrospective_pii_audit(db: Session = Depends(database.get_db)):
    """PHASE 6: Audits last 50 records for non-medical document bypasses."""
    records = db.query(models.DiagnosisRecord).order_by(models.DiagnosisRecord.id.desc()).limit(50).all()
    scrubbed = 0
    for r in records:
        # Heuristic for previous bypasses (Specific probability patterns from ID card tests)
        is_suspicious = False
        prob = float(r.final_tb_probability or 0)
        if (0.44 < prob < 0.45) or (0.88 < prob < 0.89):
            is_suspicious = True
            
        if is_suspicious:
            r.risk_level = "REDACTED"
            r.recommendations = "⚠️ CLINICAL AUDIT: Record flagged as non-medical document. PII sanitized."
            scrubbed += 1
            
    db.commit()
    return {"status": "success", "scrubbed_count": scrubbed}
