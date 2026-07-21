import datetime
import random
from app.db import database, models

def seed():
    db = database.SessionLocal()
    
    # Use existing doctor with ID 1
    doctor_id = 1
    
    # Names for patients
    first_names = ["John", "Jane", "Michael", "Emily", "David", "Sarah", "Chris", "Anna", "Robert", "Linda"]
    last_names = ["Doe", "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Wilson"]
    
    print("Seeding database with anonymous India-specific 2026 data...")
    
    # Generate ~350 records for 2026
    start_date = datetime.datetime(2026, 1, 1)
    
    # Realistic Ni-kshay monthly distribution weights (higher in Q1-Q2)
    month_weights = [1.2, 1.3, 1.5, 1.4, 1.2, 1.1, 1.0, 0.9, 0.9, 1.0, 1.1, 1.0]
    
    records_count = 350
    
    for i in range(records_count):
        # 1. Create Anonymous Patient
        patient_code = f"P-2026-{i+1000:04d}"
        age = random.randint(18, 80)
        patient = models.Patient(
            name=patient_code,
            age=age,
            gender=random.choice(["Male", "Female"]),
            doctor_id=doctor_id
        )
        db.add(patient)
        db.commit()
        db.refresh(patient)
        
        # 2. Distribute by month weight
        month = random.choices(range(1, 13), weights=month_weights)[0]
        day = random.randint(1, 28)
        record_date = datetime.datetime(2026, month, day)
        
        # 3. Clinical Data
        cough = random.randint(0, 5)
        fever = 1 if (cough > 2 and random.random() > 0.4) else 0
        weight_loss = 1 if (fever == 1 and random.random() > 0.4) else 0
        night_sweats = 1 if (weight_loss == 1 and random.random() > 0.3) else 0
        
        # Lab Data (New fields)
        sputum = random.choices([0, 1, 2], weights=[0.2, 0.4, 0.4])[0] # 0:NotDone, 1:Pos, 2:Neg
        genexpert = random.choices([0, 1, 2], weights=[0.2, 0.3, 0.5])[0]
        
        # 4. Probabilities logic
        cnn_prob = random.uniform(0.1, 0.95)
        # Clinical score boosted by lab data
        clin_score = (cough * 0.1) + (fever * 0.2) + (weight_loss * 0.2)
        if sputum == 1: clin_score += 0.4
        if genexpert == 1: clin_score += 0.5
        clin_prob = min(0.99, clin_score + random.uniform(0.0, 0.1))
        
        final_prob = (cnn_prob * 0.5) + (clin_prob * 0.5)
        
        if final_prob > 0.7 or sputum == 1 or genexpert == 1:
            risk = "High"
            rec = "Confirmed TB case. Immediate treatment initiation per NTEP guidelines."
        elif final_prob > 0.4:
            risk = "Medium"
            rec = "Presumptive TB. Repeat Sputum/GeneXpert in 2 weeks."
        else:
            risk = "Low"
            rec = "TB not detected. Monitor for persistent symptoms."
            
        record = models.DiagnosisRecord(
            patient_id=patient.id,
            cough_duration_weeks=cough,
            fever=fever,
            weight_loss=weight_loss,
            night_sweats=night_sweats,
            sputum_test=sputum,
            genexpert_test=genexpert,
            cnn_probability=cnn_prob,
            clinical_probability=clin_prob,
            final_tb_probability=final_prob,
            risk_level=risk,
            recommendations=rec,
            date=record_date
        )
        db.add(record)
        
    db.commit()
    print(f"Successfully seeded {records_count} anonymous records with India-style trends.")
    db.close()

if __name__ == "__main__":
    seed()
