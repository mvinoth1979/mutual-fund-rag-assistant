
import sys
from pathlib import Path
import logging

# Setup project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Imports
from phase_1_query_sanitization.phase_1_runner import run_phase_1
from phase_2_corpus_retrieval.phase_2_orchestrator import run_phase_2

# Configure logging
logging.basicConfig(level=logging.INFO)

def debug_query(query):
    print(f"\nDEBUGGING QUERY: {query}")
    print("="*50)
    
    # Phase 1
    p1 = run_phase_1(query)
    print(f"Phase 1 Sanitized: {p1['sanitized_query']}")
    
    # Phase 2
    p2 = run_phase_2(p1['sanitized_query'])
    print(f"Phase 2 Results: {len(p2.filtered_candidates)} candidates found.")
    
    if p2.resolved:
        print(f"Resolved Funds: {p2.resolved.mentioned_funds}")
        print(f"Fact Type: {p2.resolved.fact_type}")
    
    for i, c in enumerate(p2.filtered_candidates):
        print(f"[{i}] DOC_ID: {c.doc_id} | Type: {c.chunk_type} | Sim: {c.similarity}")
        # print(f"    Text: {c.text[:100]}...")

if __name__ == "__main__":
    debug_query("Expense Ratio of Axis Large Cap Fund")
