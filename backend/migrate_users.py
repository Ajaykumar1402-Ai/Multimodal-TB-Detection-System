import sqlite3
import os

db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tb_system.db")
if not os.path.exists(db_path):
    print(f"[-] Database not found at {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("PRAGMA table_info(users)")
columns = [row[1] for row in cursor.fetchall()]
print("Current columns in users table:", columns)

migrated = False

if "created_at" not in columns:
    print("[*] Column 'created_at' is missing in 'users' table. Migrating...")
    cursor.execute("ALTER TABLE users ADD COLUMN created_at DATETIME")
    migrated = True

if "updated_at" not in columns:
    print("[*] Column 'updated_at' is missing in 'users' table. Migrating...")
    cursor.execute("ALTER TABLE users ADD COLUMN updated_at DATETIME")
    migrated = True

if migrated:
    conn.commit()
    print("SUCCESS: SQLite 'users' table successfully migrated and synchronized with models!")
else:
    print("[*] 'users' table schema is already up to date.")

conn.close()
