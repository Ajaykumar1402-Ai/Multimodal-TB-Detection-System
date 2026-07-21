from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .api import auth, patients, inference
from .db import database, models
import os

# Create all database tables
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(
    title="Multimodal TB Detection System API",
    description="API for Tuberculosis predicting using Image and Clinical Data",
    version="1.0.0"
)

# CORS config
origins = [
    "http://localhost:5173", # Vite default
    "http://127.0.0.1:5173",
    "*" # For hackathon demo purposes
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Welcome to the Multimodal TB Detection System API"}

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(patients.router, prefix="/api/patients", tags=["patients"])
app.include_router(inference.router, prefix="/api/inference", tags=["inference"])

if not os.path.exists("./reports"):
    os.makedirs("./reports")
app.mount("/reports", StaticFiles(directory="./reports"), name="reports")
