"""
Root launcher — starts the FastAPI backend with uvicorn.
Run from project root:  python Main.py
"""
import subprocess
import sys
import os

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), "backend"))
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
        check=True,
    )
