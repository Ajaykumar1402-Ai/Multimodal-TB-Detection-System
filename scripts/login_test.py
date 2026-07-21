import requests
import time
import sys

BASE_URL = "http://localhost:8000/api/auth/login"
EMAIL = "ajaykumar348448@gmail.com"
PASSWORD = "your-secure-password-here"

def try_login(attempt=1, max_attempts=5, delay=3):
    try:
        response = requests.post(BASE_URL, json={"email": EMAIL, "password": PASSWORD}, timeout=10)
        if response.status_code == 200:
            print("[SUCCESS] Login succeeded. Token:", response.json().get('access_token'))
            return True
        else:
            print(f"[FAIL] Attempt {attempt}: HTTP {response.status_code} – {response.text}")
    except Exception as e:
        print(f"[ERROR] Attempt {attempt}: {e}")
    if attempt < max_attempts:
        time.sleep(delay)
        return try_login(attempt + 1, max_attempts, delay)
    return False

if __name__ == "__main__":
    print("[INFO] Starting login test against", BASE_URL)
    success = try_login()
    sys.exit(0 if success else 1)
