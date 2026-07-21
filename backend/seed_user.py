import sys
import os

# Append the current directory so Python can resolve app imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.db.database import SessionLocal, engine
from app.db.models import Base, User
from app.api.auth_utils import get_password_hash

def seed_user():
    # Make sure all tables are initialized
    print("[1/3] Synchronizing database tables...")
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        email = "ajaykumar348448@gmail.com"
        password = "YourPassword123"
        full_name = "Dr. Sobika"

        print(f"[2/3] Checking user presence: {email}...")
        existing_user = db.query(User).filter(User.email == email.lower().strip()).first()

        if existing_user:
            print(f"[*] User {email} already exists inside the database.")
            print("[*] Updating secure keyphrase to guarantee active access...")
            existing_user.hashed_password = get_password_hash(password)
            existing_user.full_name = full_name
            db.commit()
            print("SUCCESS: Keyphrase updated successfully!")
        else:
            print(f"[3/3] Creating clinical account for {full_name}...")
            new_user = User(
                email=email.lower().strip(),
                hashed_password=get_password_hash(password),
                full_name=full_name
            )
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            print("SUCCESS: User created successfully inside database!")
        
        print("\n" + "="*50)
        print("CLINICAL SEED COMPLETE")
        print(f"   Email: {email}")
        print(f"   Name:  {full_name}")
        print("   Status: Ready for login at /login")
        print("="*50)

    except Exception as e:
        print(f"ERROR: Failed to seed user: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    seed_user()
