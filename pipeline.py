import subprocess
import sys
import os
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

def run(command, cwd=REPO_ROOT):
    print(f"\n[{datetime.now()}] Running: {command}")
    result = subprocess.run(command, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Step failed: {command}")
        sys.exit(1)

if __name__ == "__main__":
    print(f"=== Pipeline run started at {datetime.now()} ===")

    run("python code/datasets/prepare_data.py")
    run("python code/models/train_model.py")
    run("docker-compose up --build -d", cwd=os.path.join(REPO_ROOT, "code/deployment"))

    print(f"=== Pipeline run finished at {datetime.now()} ===")
