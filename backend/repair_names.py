import random
from app.db import database, models

def repair_names():
    db = database.SessionLocal()
    
    first_names = ["Aarav", "Advik", "Akash", "Arjun", "Ishaan", "Vihaan", "Pranav", "Sai", "Vivaan", "Rohan"]
    last_names = ["Sharma", "Verma", "Gupta", "Malhotra", "Mehra", "Kapoor", "Singh", "Patel", "Reddy", "Iyer"]
    
    # Find all patients with the P-2026 pattern
    patients = db.query(models.Patient).filter(models.Patient.name.like("P-2026-%")).all()
    
    print(f"Updating {len(patients)} patient names...")
    
    for p in patients:
        p.name = f"{random.choice(first_names)} {random.choice(last_names)}"
        
    db.commit()
    print("Successfully updated all patient names.")
    db.close()

if __name__ == "__main__":
    repair_names()
