
import sys
import os
import subprocess
from pathlib import Path

def run_script(script_path):
    print(f"\n>>> Running: {script_path}")
    result = subprocess.run([sys.executable, str(script_path)], capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Script {script_path} failed with return code {result.returncode}")
        return False
    return True

def run_full_ingestion():
    print("=======================================================")
    print("   Mutual Fund FAQ Assistant - Full Ingestion Pipeline")
    print("=======================================================")
    
    phases = [
        "phase_0_ingestion/0.1_fetch/fetcher.py",
        "phase_0_ingestion/0.2_extract/extractor.py",
        "phase_0_ingestion/0.3_clean_normalize/normalizer.py",
        "phase_0_ingestion/0.4_chunk/chunker.py",
        "phase_0_ingestion/0.5_embed_index/embedder.py",
        "phase_0_ingestion/0.6_validate_audit/validator.py"
    ]
    
    for phase in phases:
        script_path = Path(phase)
        if not run_script(script_path):
            print(f"\n[CRITICAL ERROR] Ingestion aborted at {phase}")
            return

    print("\n=======================================================")
    print("   Ingestion Pipeline Completed Successfully!")
    print("=======================================================")

if __name__ == "__main__":
    run_full_ingestion()
