import os
import sys

# Ensure we can import from the app directory
sys.path.append(os.getcwd())

from app.db import database, models
from app.services import pdf_service

def repair():
    db = database.SessionLocal()
    records = db.query(models.DiagnosisRecord).all()
    print(f"Generating missing PDFs for {len(records)} records...")
    
    for r in records:
        record_data = {
            "patient_id": r.patient_id,
            "age": r.patient.age,
            "doctor_email": "demo@tbvision.com",
            "cough_duration_weeks": r.cough_duration_weeks,
            "fever": r.fever,
            "weight_loss": r.weight_loss,
            "night_sweats": r.night_sweats,
            "sputum_test": r.sputum_test,
            "genexpert_test": r.genexpert_test,
            "final_probability": r.final_tb_probability,
            "risk_level": r.risk_level,
            "recommendations": r.recommendations,
            "cnn_probability": r.cnn_probability,
            "clinical_probability": r.clinical_probability,
            "affected_regions": [],
        }
        # Safe name logic matches pdf_service.py
        try:
            pdf_service.generate_pdf_report(r.patient.name, record_data, r.id)
        except Exception as e:
            print(f"Error generating for record {r.id}: {e}")
            
    db.close()
    print("Repair complete. All historical PDFs have been generated.")

if __name__ == "__main__":
    repair()
