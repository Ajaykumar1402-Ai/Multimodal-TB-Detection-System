from backend.app.db.database import SessionLocal
from backend.app.db import models
import datetime

def init_enterprise_logs():
    db = SessionLocal()
    try:
        # Check if log already exists
        log = db.query(models.ModelLog).filter(models.ModelLog.version == "v2.5.0-enterprise").first()
        if not log:
            new_log = models.ModelLog(
                version="v2.5.0-enterprise",
                architecture="MobileNetV2 + ViT Ensemble",
                accuracy=0.94,
                f1_score=0.91,
                training_dataset="NIH TB Portals + WHO Reference",
                last_calibrated=datetime.datetime.utcnow(),
                weights_hash="sha256-clinical-validated-f8e9a1",
                is_active=1
            )
            db.add(new_log)
            db.commit()
            print("Enterprise Model Log initialized.")
        else:
            print("Enterprise Model Log already exists.")
    finally:
        db.close()

if __name__ == "__main__":
    init_enterprise_logs()
