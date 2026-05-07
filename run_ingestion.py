
import sys
import os
import subprocess
from pathlib import Path

# Get the absolute project root
PROJECT_ROOT = Path(__file__).resolve().parent

def run_script(script_path):
    abs_path = PROJECT_ROOT / script_path
    print(f"\n>>> Running: {abs_path}")
    
    # Run with PROJECT_ROOT as CWD, capture output to log it
    result = subprocess.run(
        [sys.executable, str(abs_path)], 
        capture_output=True, 
        text=True,
        cwd=str(PROJECT_ROOT)
    )
    
    # Log stdout and stderr
    if result.stdout:
        print(f"[STDOUT] {script_path}:\n{result.stdout}")
    if result.stderr:
        print(f"[STDERR] {script_path}:\n{result.stderr}")
    
    if result.returncode != 0:
        print(f"[ERROR] Script {script_path} failed with return code {result.returncode}")
        return False
    return True

def run_full_ingestion():
    print("=======================================================")
    print("   Mutual Fund FAQ Assistant - Full Ingestion Pipeline")
    print("=======================================================")
    
    # Ensure data directories exist
    data_dirs = [
        "data/0_raw_html",
        "data/1_extracted_facts",
        "data/2_cleaned_facts",
        "data/3_chunks",
        "data/4_embeddings",
        "data/5_structured_facts"
    ]
    for d in data_dirs:
        dir_path = PROJECT_ROOT / d
        if not dir_path.exists():
            print(f"> Creating directory: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
    
    phases = [
        "phase_0_ingestion/0.1_fetch/fetcher.py",
        "phase_0_ingestion/0.2_extract/extractor.py",
        "phase_0_ingestion/0.3_clean_normalize/normalizer.py",
        "phase_0_ingestion/0.4_chunk/chunker.py",
        "phase_0_ingestion/0.5_embed_index/embedder.py",
        "phase_0_ingestion/0.6_validate_audit/validator.py"
    ]
    
    for phase in phases:
        if not run_script(phase):
            error_msg = f"Ingestion failed at {phase}"
            print(f"\n[CRITICAL ERROR] {error_msg}")
            # Raise exception so the API knows exactly where it failed
            raise Exception(error_msg)

    print("\n=======================================================")
    print("   Ingestion Pipeline Completed Successfully!")
    print("=======================================================")

if __name__ == "__main__":
    run_full_ingestion()
