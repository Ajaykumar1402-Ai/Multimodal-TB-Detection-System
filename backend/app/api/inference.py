from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from sqlalchemy.orm import Session
from ..db import database, models
from .auth_utils import verify_password
from ..services import ml_pipeline, pdf_service, email_service
import json
import os
import csv
import datetime
import filetype

router = APIRouter()

# PHASE 1 SAFETY LOCK: Set to True only when clinical validation is 100% confirmed.
VALIDATION_CONFIRMED = True

@router.post("/predict")
async def predict_tb(
    background_tasks: BackgroundTasks,
    patient_id: int = Form(...),
    patient_name: str = Form(...),
    doctor_email: str = Form(...),
    age: int = Form(45),
    date_of_birth: str = Form(None),
    cough_duration_weeks: int = Form(0),
    fever: int = Form(0),
    weight_loss: int = Form(0),
    night_sweats: int = Form(0),
    sputum_test: int = Form(0),
    genexpert_test: int = Form(0),
    no_symptoms: int = Form(0),
    xray_image: UploadFile = File(...),
    db: Session = Depends(database.get_db)
):
    # TEMPORARY SAFETY LOCK (MASTER FIX PHASE 1)
    if not VALIDATION_CONFIRMED:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Service temporarily unavailable",
                "message": "System undergoing safety validation. Please try again shortly.",
                "code": "SYSTEM_LOCKED"
            }
        )

    try:
        # 0. Server-side Validation
        if not no_symptoms and not (fever or weight_loss or night_sweats or cough_duration_weeks > 0):
             raise HTTPException(status_code=400, detail="Clinical validation failed. Please select at least one symptom or 'No symptoms'.")

        # Check file size (20 MB limit)
        MAX_FILE_SIZE = 20 * 1024 * 1024
        contents = await xray_image.read()
        if len(contents) > MAX_FILE_SIZE:
             raise HTTPException(status_code=400, detail="Invalid file. Please upload a JPEG, PNG, or DICOM image under 20 MB.")
        
        # Check file type
        kind = filetype.guess(contents)
        allowed_types = ['image/jpeg', 'image/png', 'application/dicom']
        # Also check file extension for DICOM as filetype might not guess it correctly without magic bytes
        if kind is None and not xray_image.filename.lower().endswith('.dcm'):
            raise HTTPException(status_code=400, detail="Invalid file. Please upload a JPEG, PNG, or DICOM image under 20 MB.")
        
        if kind and kind.mime not in allowed_types and not xray_image.filename.lower().endswith('.dcm'):
            raise HTTPException(status_code=400, detail="Invalid file. Please upload a JPEG, PNG, or DICOM image under 20 MB.")

        # Reset cursor after reading
        # Note: Since contents is already in memory, we use it directly below
        
        # 1. Check patient exists
        patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
        if not patient:
            patient = models.Patient(id=patient_id, name=patient_name, age=age, date_of_birth=date_of_birth, gender="N/A")
            db.add(patient)
        else:
            patient.name = patient_name
            patient.age = age # Update if exists
            if date_of_birth:
                patient.date_of_birth = date_of_birth
        db.commit()
        db.refresh(patient)

        # 2. Process image (Gated by Authenticity Validator)
        image_bytes = contents 
        filename = xray_image.filename
        print(f"VALIDATOR CALLED: {filename}, {len(image_bytes)} bytes")
        print(f"[PIPELINE] Starting inference for patient {patient_id}. Image size: {len(image_bytes)} bytes")
        
        # This call inside process_xray_ensemble will trigger validate_xray_image
        ensemble_results = ml_pipeline.process_xray_ensemble(image_bytes, filename=filename)
        
        print(f"[PIPELINE] Authenticity confirmed for {filename}. AI Score: {ensemble_results.get('ensemble_total')}")
        cnn_prob = ensemble_results["ensemble_total"]
        
        # 2.5 Generate MedSAM segmentation with region classification
        medsam_result = ml_pipeline.generate_medsam_segmentation(image_bytes, cnn_prob=cnn_prob)
        medsam_mask_url = medsam_result.get("mask_url")
        affected_regions = medsam_result.get("regions", [])
        
        # 3. Process clinical data (for input)
        clinical_data = {
            "cough_duration_weeks": cough_duration_weeks,
            "fever": fever,
            "weight_loss": weight_loss,
            "night_sweats": night_sweats,
            "sputum_test": sputum_test,
            "genexpert_test": genexpert_test,
            "no_symptoms": no_symptoms
        }

        # --- SINGLE ENTRY POINT FOR INFERENCE ---
        inference_result = ml_pipeline.process_input(image_bytes, filename, clinical_data)

        if not inference_result.get("validation_passed"):
            raise HTTPException(
                status_code=422,
                detail=inference_result
            )

        # 4. Extract data for DB storage and response
        final_prob = inference_result["probability_mean"]
        risk_level = inference_result["risk_level"]
        recommendations = inference_result["recommendations"]
        cnn_prob = inference_result["cnn_probability"]
        clin_prob = inference_result["clinical_probability"]
        affected_regions = inference_result["affected_regions"]
        
        record_data = {
            **clinical_data,
            "patient_id": patient_id,
            "age": age,
            "date_of_birth": patient.date_of_birth,
            "final_probability": final_prob,
            "risk_level": risk_level,
            "recommendations": recommendations,
            "cnn_probability": cnn_prob,
            "clinical_probability": clin_prob,
            "affected_regions": affected_regions,
            "doctor_email": doctor_email,
            "confidence_interval": inference_result["confidence_interval"],
            "uncertainty": inference_result["uncertainty"]
        }
        
        # 5. Save Record (Enterprise Standard)
        new_record = models.DiagnosisRecord(
            patient_id=patient.id,
            visit_date=datetime.datetime.utcnow(),
            cough_duration_weeks=cough_duration_weeks,
            fever=fever,
            weight_loss=weight_loss,
            night_sweats=night_sweats,
            no_symptoms=no_symptoms,
            sputum_test=sputum_test,
            genexpert_test=genexpert_test,
            xray_image_path=xray_image.filename,
            
            # AI Telemetry
            model_version="v2.5.0-enterprise",
            inference_latency_ms=0.0,
            confidence_interval=str(inference_result["confidence_interval"]),
            
            # Results
            cnn_probability=cnn_prob,
            clinical_probability=clin_prob,
            final_tb_probability=final_prob,
            risk_level=risk_level,
            recommendations=recommendations,
            
            # Communication
            is_email_notified=1, 
            notified_at=datetime.datetime.utcnow()
        )
        db.add(new_record)
        db.commit()
        db.refresh(new_record)
        
        # 6. Generate PDF and send email asynchronously
        pdf_path = pdf_service.generate_pdf_report(patient.name, record_data, new_record.id)
        new_record.report_path = pdf_path # Save PDF link to record
        db.commit()
        
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

        # --- STEP 4: URGENT PII WIPE ---
        def wipe_pii(data):
            allowed_patient_info = {
                "patient_name": patient_name,
                "patient_id": patient_id,
                "doctor_email": doctor_email
            }
            results = data.get("results", {})
            results.pop("extracted_text", None)
            results.pop("ocr_data", None)
            results.pop("document_content", None)
            
            return {
                "record_id": data.get("record_id"),
                "results": results,
                "patient_info": allowed_patient_info,
                "email_sent": False,
                "pdf_url": data.get("pdf_url"),
                # Root level exact matching the requested JSON format
                "prediction": data.get("prediction"),
                "probability_mean": data.get("probability_mean"),
                "confidence_interval": data.get("confidence_interval"),
                "uncertainty": data.get("uncertainty"),
                "validation_passed": data.get("validation_passed")
            }

        raw_response = {
            "record_id": new_record.id,
            "prediction": inference_result["prediction"],
            "probability_mean": inference_result["probability_mean"],
            "confidence_interval": inference_result["confidence_interval"],
            "uncertainty": inference_result["uncertainty"],
            "validation_passed": inference_result["validation_passed"],
            "results": {
                "final_prob": final_prob,
                "risk_level": risk_level,
                "recommendations": recommendations,
                "cnn_probability": cnn_prob,
                "clinical_probability": clin_prob,
                "ensemble_breakdown": inference_result.get("ensemble_breakdown"),
                "medsam_mask_url": inference_result.get("medsam_mask_url"),
                "affected_regions": affected_regions,
            },
            "email_sent": False, 
            "pdf_url": f"https://multimodal-tb-detection-system.onrender.com/api/stats/report/{new_record.id}"
        }

        return wipe_pii(raw_response)
    except HTTPException:
        raise
    except Exception as e:
        print(f"[AUDIT] SYSTEM ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference Engine Error: {str(e)}")
