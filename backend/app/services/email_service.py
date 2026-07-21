import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import os
from dotenv import load_dotenv

load_dotenv()

SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "mock_email@example.com")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "mock_password")

def send_results_email(to_email: str, patient_name: str, risk_level: str, pdf_path: str = None):
    # For hackathon/demo purposes, we will mock this if credentials aren't provided
    if SMTP_USERNAME == "mock_email@example.com":
        print(f"MOCK EMAIL SENT TO {to_email} FOR PATIENT {patient_name} with RISK {risk_level}")
        return True

    msg = MIMEMultipart()
    msg['From'] = SMTP_USERNAME
    msg['To'] = to_email
    msg['Subject'] = f"TB Detection Results - {patient_name}"

    body = f"""
    Hello Doctor,

    The AI analysis for patient {patient_name} has been completed.
    
    Overall Risk Level: {risk_level}
    
    Please find the detailed diagnostic report attached.

    Regards,
    Multimodal TB System
    """
    msg.attach(MIMEText(body, 'plain'))

    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            attach = MIMEApplication(f.read(), _subtype="pdf")
            attach.add_header('Content-Disposition', 'attachment', filename=os.path.basename(pdf_path))
            msg.attach(attach)

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False
