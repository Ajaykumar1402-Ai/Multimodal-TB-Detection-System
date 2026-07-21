from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from ..db import database, models
from .auth_utils import verify_password, get_password_hash, create_access_token
import secrets
import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

_reset_tokens: dict = {}
TOKEN_EXPIRY_MINUTES = 15

class UserCreate(BaseModel):
    email: str
    password: str
    full_name: str

class UserLogin(BaseModel):
    email: str
    password: str

class PasswordResetRequest(BaseModel):
    email: str

class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str

@router.post("/signup", status_code=status.HTTP_201_CREATED)
def signup(user: UserCreate, db: Session = Depends(database.get_db)):
    email = user.email.lower().strip()
    if len(user.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    db_user = db.query(models.User).filter(models.User.email == email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    new_user = models.User(email=email, hashed_password=hashed_password, full_name=user.full_name)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created successfully"}

@router.post("/login")
def login(user: UserLogin, db: Session = Depends(database.get_db)):
    email = user.email.lower().strip()
    try:
        try:
            db_user = db.query(models.User).filter(models.User.email == email).first()
        except Exception as db_err:
            # Check for SQLAlchemy database connection/operational issues
            err_msg = str(db_err).lower()
            logger.error(f"DATABASE CONNECTION FAILURE: {db_err}")
            
            if "connrefused" in err_msg or "connection refused" in err_msg or "operationalerror" in err_msg or "is not accepting connections" in err_msg:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Database connection failed — backend database may be starting up. Please retry in 30 seconds."
                )
            if "timeout" in err_msg or "timed out" in err_msg:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Server is starting up (database handshake timeout). Please wait 30 seconds and try again."
                )
            raise db_err
        
        if not db_user:
            logger.warning(f"LOGIN FAILURE: User {email} not found")
            raise HTTPException(status_code=401, detail="Invalid email or password")
            
        if not verify_password(user.password, db_user.hashed_password):
            logger.warning(f"LOGIN FAILURE: Invalid password for {email}")
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        access_token = create_access_token(data={"sub": db_user.email})
        logger.info(f"LOGIN SUCCESS: {email}")
        return {
            "access_token": access_token, 
            "token_type": "bearer", 
            "user_name": db_user.full_name
        }
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"DATABASE OR AUTH ERROR: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error: Login failed unexpectedly. Please try again.")

import os
import resend

resend.api_key = os.environ.get("RESEND_API_KEY")

@router.post("/request-reset")
def request_password_reset(
    payload: PasswordResetRequest,
    db: Session = Depends(database.get_db),
):
    email = payload.email.lower().strip()
    db_user = db.query(models.User).filter(models.User.email == email).first()
    
    if db_user:
        token = secrets.token_urlsafe(32)
        _reset_tokens[token] = {
            "email": email,
            "expires": datetime.datetime.utcnow() + datetime.timedelta(minutes=TOKEN_EXPIRY_MINUTES),
        }
        
        frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:5173").rstrip('/')
        reset_link = f"{frontend_url}/reset-password?token={token}"
        print(f"[DEV] Password reset for {email}: {reset_link}")
        
        try:
            if not resend.api_key:
                raise ValueError("RESEND_API_KEY is not configured on the server.")
                
            params = {
                "from": "onboarding@resend.dev",
                "to": [email],
                "subject": "TB-Vision Pro: Password Reset Request",
                "html": f"<p>Hello {db_user.full_name},</p><p>Click the link below to reset your secure keyphrase. This link will expire in {TOKEN_EXPIRY_MINUTES} minutes.</p><p><a href='{reset_link}'><strong>Reset Password</strong></a></p><p>If you did not request this, please safely ignore this email.</p>"
            }
            resend.Emails.send(params)
            
            # Note: Database audit logging disabled temporarily to ensure resilient email delivery
            
        except Exception as e:
            error_details = str(e)
            # Temporarily surface the exact error to the frontend for debugging
            raise HTTPException(status_code=500, detail=f"Resend Error: {error_details}")

            
    return {"message": "If that email is registered, reset instructions have been sent."}



@router.post("/confirm-reset")
def confirm_password_reset(
    payload: PasswordResetConfirm,
    db: Session = Depends(database.get_db),
):
    token_data = _reset_tokens.get(payload.token)
    if not token_data:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")
    if datetime.datetime.utcnow() > token_data["expires"]:
        del _reset_tokens[payload.token]
        raise HTTPException(status_code=400, detail="Reset token has expired. Please request a new one.")
    if len(payload.new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    
    email = token_data["email"]
    db_user = db.query(models.User).filter(models.User.email == email).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    db_user.hashed_password = get_password_hash(payload.new_password)
    db.commit()
    del _reset_tokens[payload.token]
    return {"message": "Password updated successfully. You can now log in."}
