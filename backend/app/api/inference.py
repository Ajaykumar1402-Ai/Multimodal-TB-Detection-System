from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from sqlalchemy.orm import Session
from ..db import database, models
from .auth_utils import verify_password
from ..services import ml_pipeline, pdf_service, email_service
import json
import os
import csv
import datetime

router = APIRouter()

@router.post("/predict")
async def predict_tb(
    background_tasks: BackgroundTasks,
    patient_id: int = Form(...),
    patient_name: str = Form(...),
    doctor_email: str = Form(...),
    age: int = Form(45),
    cough_duration_weeks: int = Form(0),
    fever: int = Form(0),
    weight_loss: int = Form(0),
    night_sweats: int = Form(0),
    xray_image: UploadFile = File(...),
    db: Session = Depends(database.get_db)
):
    # 1. Check patient exists (Auto-create for hackathon demo if missing)
    patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
    if not patient:
        patient = models.Patient(id=patient_id, name=patient_name, age=age, gender="N/A")
        db.add(patient)
    else:
        patient.name = patient_name
        patient.age = age # Update if exists
    db.commit()
    db.refresh(patient)

    # 2. Process image
    image_bytes = await xray_image.read()
    cnn_prob = ml_pipeline.process_xray_image(image_bytes)
    gradcam_path = None # Simulation
    
    # 3. Process clinical data
    clinical_data = {
        "cough_duration_weeks": cough_duration_weeks,
        "fever": fever,
        "weight_loss": weight_loss,
        "night_sweats": night_sweats
    }
    clin_prob = ml_pipeline.process_clinical_data(clinical_data)
    
    # 4. Fusion Model
    fusion_result = ml_pipeline.multimodal_fusion(cnn_prob, clin_prob)
    final_prob = fusion_result["final_prob"]
    risk_level = fusion_result["risk_level"]
    recommendations = fusion_result["recommendations"]
    
    record_data = {
        **clinical_data,
        "patient_id": patient_id,
        "age": age,
        "final_probability": final_prob,
        "risk_level": risk_level,
        "recommendations": recommendations,
        "cnn_probability": cnn_prob,
        "clinical_probability": clin_prob
    }
    
    # 5. Save Record
    new_record = models.DiagnosisRecord(
        patient_id=patient.id,
        cough_duration_weeks=cough_duration_weeks,
        fever=fever,
        weight_loss=weight_loss,
        night_sweats=night_sweats,
        xray_image_path=xray_image.filename,
        cnn_probability=cnn_prob,
        clinical_probability=clin_prob,
        final_tb_probability=final_prob,
        risk_level=risk_level,
        recommendations=recommendations
    )
    db.add(new_record)
    db.commit()
    db.refresh(new_record)
    
    # 6. Generate PDF and send email asynchronously
    pdf_path = pdf_service.generate_pdf_report(patient.name, record_data)
    background_tasks.add_task(email_service.send_results_email, doctor_email, patient.name, risk_level, pdf_path)
    
    # 7. Append to Master Excel/CSV file
    csv_file_path = "reports/patient_records.csv"
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write Header row if file evaluates as new
            writer.writerow(["Date", "Patient ID", "Name", "Age", "Cough(Weeks)", "Fever", "Weight Loss", "Night Sweats", "Risk Level", "Final Prob", "CNN Prob", "Clinical Prob"])
        # Write exact data row
        writer.writerow([
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            patient_id, patient_name, age, cough_duration_weeks, fever, weight_loss, night_sweats, 
            risk_level, f"{final_prob*100:.1f}%", f"{cnn_prob*100:.1f}%", f"{clin_prob*100:.1f}%"
        ])

    return {
        "record_id": new_record.id,
        "results": {
            "final_prob": final_prob,
            "risk_level": risk_level,
            "recommendations": recommendations,
            "cnn_probability": cnn_prob,
            "clinical_probability": clin_prob,
            "gradcam_url": None
        },
        "email_sent": False, # Will show 'Email Queued' in UI
        "pdf_url": f"http://localhost:8000/reports/{os.path.basename(pdf_path)}" if pdf_path else None
    }
