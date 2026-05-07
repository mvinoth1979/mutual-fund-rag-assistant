import sys
import os
import sqlite3
import json
from pathlib import Path
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Silence most logs
logging.getLogger().setLevel(logging.ERROR)

# Imports from phases
from phase_1_query_sanitization.phase_1_runner import run_phase_1
from phase_2_corpus_retrieval.phase_2_orchestrator import run_phase_2
from phase_3_context_assembly.pipeline import ContextAssemblyPipeline
from phase_3_context_assembly.models import Chunk
from phase_4_response_generation.generator import ResponseGenerator
from phase_5_compliance_validation.pipeline import CompliancePipeline

def get_whitelist():
    db_path = PROJECT_ROOT / "data" / "5_structured_facts" / "facts.db"
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT source_url FROM structured_facts")
            return [row[0] for row in cursor.fetchall()]
    except Exception:
        return []

def run_query(query, pipeline_phase_3, generator_phase_4, pipeline_phase_5):
    # PHASE 1
    p1_result = run_phase_1(query)
    if p1_result["blocked_by_pii"] or p1_result["terminal_response"]:
        term = p1_result["terminal_response"]
        if isinstance(term, dict):
            return term.get("text", "BLOCKED BY PII")
        return term or "BLOCKED BY PII"
    
    sanitized_query = p1_result["sanitized_query"]

    # PHASE 2
    p2_result = run_phase_2(sanitized_query)
    if p2_result.advisory_triggered:
        return p2_result.advisory_reason or "ADVISORY TRIGGERED"
        
    candidates = p2_result.filtered_candidates
    if not candidates:
        return "No relevant information found in the dataset."

    # Map to Phase 3 Chunk model
    phase_3_chunks = [Chunk(
        chunk_id=c.chunk_id,
        doc_id=c.doc_id,
        source_url=c.source_url,
        chunk_type=c.chunk_type,
        text=c.text
    ) for c in candidates]

    # PHASE 3
    context_string, source_url, doc_id = pipeline_phase_3.execute(phase_3_chunks)

    # PHASE 4
    p4_response = generator_phase_4.generate_response(context_string, sanitized_query)
    if p4_response.status != "SUCCESS":
        return f"ERROR: {p4_response.text}"

    # PHASE 5
    p5_result = pipeline_phase_5.validate(
        raw_response=p4_response.text,
        source_context=context_string,
        source_url=source_url
    )

    return p5_result.response

def main():
    queries_path = PROJECT_ROOT / "scratch" / "test_queries.json"
    with open(queries_path, "r") as f:
        test_queries = json.load(f)

    whitelist = get_whitelist()
    pipeline_phase_3 = ContextAssemblyPipeline(whitelist=whitelist, max_tokens=2000)
    generator_phase_4 = ResponseGenerator()
    
    banned_phrases_path = os.getenv("BANNED_PHRASES_PATH", "./phase_5_compliance_validation/5.1_advisory_detection/banned_phrases.json")
    banned_phrases_path = PROJECT_ROOT / banned_phrases_path.replace("./", "")
    pipeline_phase_5 = CompliancePipeline(banned_phrases_path=str(banned_phrases_path))

    results = []
    print(f"Executing {len(test_queries)} test queries...")

    for i, tq in enumerate(test_queries):
        query_text = tq["query"]
        print(f"[{i+1}/{len(test_queries)}] Processing: {query_text[:50]}...")
        actual = run_query(query_text, pipeline_phase_3, generator_phase_4, pipeline_phase_5)
        
        results.append({
            "category": tq["category"],
            "query": query_text,
            "expected": tq["expected"],
            "actual": actual
        })

    # Generate Report
    report_path = PROJECT_ROOT / "Docs" / "evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# RAG System Evaluation Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Queries:** {len(results)}\n\n")
        
        f.write("## Summary\n\n")
        categories = ["CORRECT", "PARTIAL", "ARBITRARY"]
        for cat in categories:
            cat_results = [r for r in results if r["category"] == cat]
            f.write(f"- **{cat}:** {len(cat_results)} queries\n")
        
        f.write("\n## Comparison Table\n\n")
        f.write("| # | Category | Query | Expected | Actual | Deviation |\n")
        f.write("|---|----------|-------|----------|--------|-----------|\n")
        
        for i, r in enumerate(results):
            # Simple deviation check: check if actual is a refusal for arbitrary, or contains key words for correct
            deviation = "None"
            if r["category"] == "ARBITRARY":
                if "advise" not in r["actual"].lower() and "cannot" not in r["actual"].lower() and "sorry" not in r["actual"].lower() and "only" not in r["actual"].lower():
                    deviation = "HALLUCINATION / FAILED REFUSAL"
            elif r["category"] == "CORRECT":
                if "No relevant information" in r["actual"]:
                    deviation = "RETRIEVAL FAILURE"
            
            # Escape pipes in content
            q_esc = r["query"].replace("|", "\\|")
            exp_esc = r["expected"].replace("|", "\\|")
            act_esc = r["actual"].replace("|", "\\|").replace("\n", " ")
            
            f.write(f"| {i+1} | {r['category']} | {q_esc} | {exp_esc} | {act_esc} | {deviation} |\n")

    print(f"Report generated at: {report_path}")

if __name__ == "__main__":
    main()
