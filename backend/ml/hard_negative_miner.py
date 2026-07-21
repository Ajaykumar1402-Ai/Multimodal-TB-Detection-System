import os
import sqlite3
import shutil
import uuid
import datetime
from pathlib import Path

# Paths
DB_PATH = Path(os.path.dirname(__file__)).parent / 'data' / 'hard_negatives.db'
LOGGED_UPLOADS_DIR = Path(os.path.dirname(__file__)).parent / 'data' / 'production_logs'
GUARD_DATASET_ROOT = Path(os.path.dirname(__file__)).parent / 'data' / 'train_guard'

def init_db():
    """Initializes the SQLite database for tracking hard negatives."""
    LOGGED_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS hard_negatives (
            id TEXT PRIMARY KEY,
            original_path TEXT,
            logged_path TEXT,
            timestamp TEXT,
            rejection_reason TEXT,
            status TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_incorrectly_accepted_image(image_path, reason="Failed Guard AI Validation manually reported"):
    """
    Logs an image that was incorrectly accepted by the Guard AI into the Hard Negative Mining system.
    """
    init_db()
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    unique_id = uuid.uuid4().hex
    timestamp = datetime.datetime.now().isoformat()
    extension = Path(image_path).suffix
    logged_filename = f"{unique_id}{extension}"
    logged_path = LOGGED_UPLOADS_DIR / logged_filename
    
    shutil.copy2(image_path, logged_path)
    
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO hard_negatives (id, original_path, logged_path, timestamp, rejection_reason, status)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (unique_id, str(image_path), str(logged_path), timestamp, reason, 'PENDING_REVIEW'))
    conn.commit()
    conn.close()
    
    print(f"Logged hard negative: {unique_id}")
    return unique_id

def list_pending_reviews():
    """Returns a list of all hard negatives pending manual review."""
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("SELECT id, logged_path, rejection_reason, timestamp FROM hard_negatives WHERE status='PENDING_REVIEW'")
    results = cursor.fetchall()
    conn.close()
    return results

def approve_hard_negative(unique_id, category_label="hard_negatives"):
    """
    Approves a logged image and automatically moves it to the training dataset.
    """
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("SELECT logged_path FROM hard_negatives WHERE id=? AND status='PENDING_REVIEW'", (unique_id,))
    row = cursor.fetchone()
    
    if not row:
        print(f"No pending review found for ID: {unique_id}")
        conn.close()
        return False
        
    logged_path = row[0]
    
    dest_dir = GUARD_DATASET_ROOT / 'non_cxr' / 'hard_negatives'
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    dest_file = dest_dir / os.path.basename(logged_path)
    shutil.copy2(logged_path, dest_file)
    
    cursor.execute("UPDATE hard_negatives SET status='APPROVED' WHERE id=?", (unique_id,))
    conn.commit()
    conn.close()
    print(f"Approved ID: {unique_id}. Image added to {dest_dir}")
    return True

def reject_hard_negative(unique_id):
    """
    Rejects a logged image (meaning it was actually a valid CXR or shouldn't be added to negatives).
    """
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("UPDATE hard_negatives SET status='REJECTED' WHERE id=?", (unique_id,))
    conn.commit()
    conn.close()
    print(f"Rejected ID: {unique_id}.")
    return True

if __name__ == '__main__':
    init_db()
    print("Hard Negative Mining System Initialized.")
