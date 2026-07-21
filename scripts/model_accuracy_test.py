# Model accuracy and stats test script
import subprocess
import sys

# Run external evaluation script
print("[INFO] Running external model evaluation...")
subprocess.run([sys.executable, "backend/ml_training/evaluate_model.py"], check=False)

# Run quick stats from database
print("[INFO] Running test_stats for statistical check...")
subprocess.run([sys.executable, "backend/test_stats.py"], check=False)
