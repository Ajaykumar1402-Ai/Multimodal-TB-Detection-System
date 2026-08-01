from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from .api import auth, patients, inference, stats
from .db import database, models
from contextlib import asynccontextmanager
import os
import logging
import traceback
from fastapi.responses import JSONResponse

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from sqlalchemy import inspect
from sqlalchemy.sql import text

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== TB-VISION PRO: CLINICAL ENGINE STARTING ===")
    
    # 1. Base table synchronization
    try:
        models.Base.metadata.create_all(bind=database.engine)
        logger.info("Database tables verified/created successfully.")
    except Exception as e:
        logger.error(f"Database init warning (non-fatal): {e}")

    # 2. Self-healing migration for missing table columns
    try:
        inspector = inspect(database.engine)
        table_names = inspector.get_table_names()
        is_sqlite = "sqlite" in str(database.engine.url)
        alter_time_type = "DATETIME" if is_sqlite else "TIMESTAMP"
        
        # A. Users table migrations
        if "users" in table_names:
            columns = [c["name"] for c in inspector.get_columns("users")]
            with database.engine.connect() as conn:
                trans = conn.begin()
                try:
                    migrated = False
                    if "created_at" not in columns:
                        logger.info("MIGRATION: Adding created_at column to users table...")
                        conn.execute(text(f"ALTER TABLE users ADD COLUMN created_at {alter_time_type}"))
                        migrated = True
                    if "updated_at" not in columns:
                        logger.info("MIGRATION: Adding updated_at column to users table...")
                        conn.execute(text(f"ALTER TABLE users ADD COLUMN updated_at {alter_time_type}"))
                        migrated = True
                    trans.commit()
                    if migrated:
                        logger.info("MIGRATION: Successfully synchronized users table columns!")
                except Exception as alter_err:
                    trans.rollback()
                    logger.error(f"MIGRATION ERROR: Alter table users failed: {alter_err}")

        # B. Patients table migrations
        if "patients" in table_names:
            columns = [c["name"] for c in inspector.get_columns("patients")]
            with database.engine.connect() as conn:
                trans = conn.begin()
                try:
                    migrated = False
                    if "date_of_birth" not in columns:
                        logger.info("MIGRATION: Adding date_of_birth column to patients table...")
                        conn.execute(text("ALTER TABLE patients ADD COLUMN date_of_birth VARCHAR(50)"))
                        migrated = True
                    if "updated_at" not in columns:
                        logger.info("MIGRATION: Adding updated_at column to patients table...")
                        conn.execute(text(f"ALTER TABLE patients ADD COLUMN updated_at {alter_time_type}"))
                        migrated = True
                    trans.commit()
                    if migrated:
                        logger.info("MIGRATION: Successfully synchronized patients table columns!")
                except Exception as alter_err:
                    trans.rollback()
                    logger.error(f"MIGRATION ERROR: Alter table patients failed: {alter_err}")

        # C. Diagnosis Records table migrations
        if "diagnosis_records" in table_names:
            columns = [c["name"] for c in inspector.get_columns("diagnosis_records")]
            with database.engine.connect() as conn:
                trans = conn.begin()
                try:
                    migrated = False
                    
                    # Special check: Rename/copy 'date' to 'visit_date'
                    if "visit_date" not in columns:
                        logger.info("MIGRATION: Adding visit_date column to diagnosis_records table...")
                        conn.execute(text(f"ALTER TABLE diagnosis_records ADD COLUMN visit_date {alter_time_type}"))
                        migrated = True
                        
                    # Other missing fields
                    cols_to_add = {
                        "no_symptoms": "INTEGER DEFAULT 0",
                        "model_version": "VARCHAR(100) DEFAULT 'v2.5.0-enterprise'",
                        "inference_latency_ms": "FLOAT DEFAULT 0.0",
                        "confidence_interval": "VARCHAR(50)",
                        "grad_cam_heatmap": "VARCHAR(255)",
                        "report_path": "VARCHAR(255)",
                        "is_email_notified": "INTEGER DEFAULT 0",
                        "notified_at": alter_time_type,
                        "created_at": alter_time_type,
                        "updated_at": alter_time_type
                    }
                    
                    for col_name, col_def in cols_to_add.items():
                        if col_name not in columns:
                            logger.info(f"MIGRATION: Adding {col_name} column to diagnosis_records table...")
                            conn.execute(text(f"ALTER TABLE diagnosis_records ADD COLUMN {col_name} {col_def}"))
                            migrated = True
                            
                    trans.commit()
                    
                    # Post-migration sync: If visit_date was added and we have historical 'date' column
                    if "date" in columns:
                        with database.engine.connect() as conn2:
                            trans2 = conn2.begin()
                            try:
                                conn2.execute(text("UPDATE diagnosis_records SET visit_date = date WHERE visit_date IS NULL AND date IS NOT NULL"))
                                trans2.commit()
                                logger.info("MIGRATION: Copied legacy 'date' to 'visit_date' in diagnosis_records table.")
                            except Exception as copy_err:
                                trans2.rollback()
                                logger.warning(f"MIGRATION WARNING: Copying legacy 'date' values failed: {copy_err}")
                                
                    if migrated:
                        logger.info("MIGRATION: Successfully synchronized diagnosis_records table columns!")
                except Exception as alter_err:
                    trans.rollback()
                    logger.error(f"MIGRATION ERROR: Alter table diagnosis_records failed: {alter_err}")
                    
    except Exception as inspect_err:
        logger.error(f"MIGRATION ERROR: Database schema inspection failed: {inspect_err}")


    # 3. Dynamic clinical user seeding
    try:
        from app.api.auth_utils import get_password_hash
        db_sess = database.SessionLocal()
        try:
            email = "ajaykumar348448@gmail.com"
            existing = db_sess.query(models.User).filter(models.User.email == email).first()
            if not existing:
                logger.info(f"SEEDING: Clinician account {email} not found. Seeding default profile...")
                default_user = models.User(
                    email=email,
                    hashed_password=get_password_hash("your-secure-password-here"),
                    full_name="Dr. Sobika"
                )
                db_sess.add(default_user)
                db_sess.commit()
                logger.info("SEEDING: Default clinician account successfully registered!")
            else:
                logger.info(f"SEEDING: Clinician account {email} is active.")
        except Exception as seed_err:
            logger.error(f"SEEDING ERROR: Seeding clinician failed: {seed_err}")
        finally:
            db_sess.close()
    except Exception as import_err:
        logger.error(f"SEEDING ERROR: Auth helper imports failed: {import_err}")

    # 4. GPU & Model Warmup
    try:
        import torch
        logger.info("GPU WARMUP: Initializing GPU and loading ML models for warmup...")
        from app.services.ml_loader import _load_pytorch_model
        from app.services.ml_pipeline import get_ocr_reader, get_cxr_classifier
        
        # Load DenseNet-121
        _load_pytorch_model()
        # Load EasyOCR
        get_ocr_reader()
        # Load MobileNetV2 Gate
        get_cxr_classifier()
        
        # Dummy pass for CUDA warmup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            logger.info(f"GPU WARMUP: Running dummy forward pass on GPU: {torch.cuda.get_device_name(0)}")
            dummy_input = torch.zeros(1, 3, 224, 224).to(device)
            
            # Warmup primary detector model
            from app.services.ml_loader import _pt_model
            if _pt_model is not None:
                model, _ = _pt_model
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        _ = model(dummy_input)
            
            # Warmup gate classifier
            from app.services.ml_pipeline import _cxr_gate_classifier
            if _cxr_gate_classifier is not None:
                gate_model, _ = _cxr_gate_classifier
                with torch.no_grad():
                    _ = gate_model(dummy_input)
            logger.info("GPU WARMUP: Warmup completed successfully.")
        else:
            logger.info("CPU WARMUP: Warmup completed.")
    except Exception as warmup_err:
        logger.error(f"WARMUP ERROR (non-fatal): {warmup_err}")

    yield
    logger.info("=== SHUTDOWN: CLEANING UP ===")

app = FastAPI(
    title="Multimodal TB Detection System API",
    description="API for Tuberculosis detection using chest X-ray images and clinical data.",
    version="3.1.0",
    lifespan=lifespan,
)

# CORS config
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://multimodal-tb-detection-system.vercel.app",
    "https://tb-vision-pro.vercel.app"
]

frontend_url = os.getenv("FRONTEND_URL")
if frontend_url:
    origins.append(frontend_url.strip().rstrip('/'))

extra_origin = os.getenv("EXTRA_FRONTEND_ORIGIN")
if extra_origin:
    origins.append(extra_origin.strip().rstrip('/'))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global Exception: {exc}")
    logger.error(traceback.format_exc())
    
    response = JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__}
    )
    
    # Manually add CORS headers to error responses
    origin = request.headers.get("origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
    else:
        response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    
    return response

@app.get("/api/ping")
def ping(db: Session = Depends(database.get_db)):
    try:
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        db_type = "sqlite" if "sqlite" in str(database.engine.url) else "postgresql"
        return {
            "status": "alive", 
            "database": "connected",
            "db_type": db_type,
            "primary_db_error": database.PRIMARY_DB_ERROR,
            "timestamp": os.getenv("RENDER_GIT_COMMIT", "local")
        }
    except Exception as e:
        logger.error(f"Health Check Failed: {str(e)}")
        return {
            "status": "degraded",
            "database": "disconnected",
            "error": str(e)
        }

@app.get("/")
def root():
    return {
        "message": "Multimodal TB Detection System API",
        "version": "3.1.0",
        "status": "operational",
    }

@app.get("/health", tags=["health"])
async def health_check():
    """Detailed health check for Render / Vercel keep-alive cron."""
    import torch
    from .services.ml_loader import _pt_model, _pt_loaded
    from datetime import datetime
    model_loaded = (_pt_model is not None or _pt_loaded)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    cuda_avail = torch.cuda.is_available()
    return {
        "status": "healthy",
        "gpu": gpu_name,
        "cuda": cuda_avail,
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/wake", tags=["health"])
async def wake():
    """Lightweight endpoint to confirm the server is awake."""
    return {"awake": True}

app.include_router(auth.router,     prefix="/api/auth",      tags=["auth"])
app.include_router(patients.router, prefix="/api/patients",  tags=["patients"])
app.include_router(inference.router,prefix="/api/inference", tags=["inference"])
app.include_router(stats.router,    prefix="/api/stats",     tags=["stats"])

os.makedirs("./reports", exist_ok=True)
app.mount("/reports", StaticFiles(directory="./reports"), name="reports")
