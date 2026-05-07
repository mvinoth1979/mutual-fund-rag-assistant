import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Setup paths and env
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv()

from phase_1_query_sanitization.phase_1_runner import run_phase_1
from phase_2_corpus_retrieval.phase_2_orchestrator import run_phase_2
from phase_3_context_assembly.pipeline import ContextAssemblyPipeline
from phase_4_response_generation.generator import ResponseGenerator
from phase_5_compliance_validation.pipeline import CompliancePipeline

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("api")

app = FastAPI(title="Mutual Fund RAG Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    text: str
    source_url: Optional[str] = None
    footer_date: str
    terminal_state: str

def load_whitelisted_urls() -> list[str]:
    """Load URLs dynamically from SourceWebsites.md."""
    source_file = PROJECT_ROOT / "SourceWebsites.md"
    urls = []
    if source_file.exists():
        with open(source_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("- "):
                    url = line[2:].strip()
                    if url.startswith("http"):
                        urls.append(url)
    return urls

# Config
WHITELISTED_URLS = os.getenv("WHITELISTED_URLS", "").split(",")
if not WHITELISTED_URLS or WHITELISTED_URLS == [""]:
    WHITELISTED_URLS = load_whitelisted_urls()

# Global pipelines
context_pipeline = ContextAssemblyPipeline(whitelist=WHITELISTED_URLS)
generation_pipeline = ResponseGenerator()
banned_phrases_path = str(PROJECT_ROOT / "phase_5_compliance_validation" / "5.1_advisory_detection" / "banned_phrases.json")
compliance_pipeline = CompliancePipeline(banned_phrases_path=banned_phrases_path)

@app.get("/api")
async def health_check():
    return {"status": "ok", "message": "Mutual Fund RAG Assistant API is running"}

@app.get("/api/debug-data")
@app.get("/debug-data")
async def debug_data():
    results = {}
    # List files in PROJECT_ROOT recursively, but limit to avoid huge output
    for item in PROJECT_ROOT.rglob("*"):
        if item.is_file():
            rel_path = str(item.relative_to(PROJECT_ROOT))
            # Only show relevant data files or config files
            if "data/" in rel_path or "api/" in rel_path or ".json" in rel_path:
                results[rel_path] = f"{item.stat().st_size} bytes"
    
    # Also check specific critical paths
    manifest_path = PROJECT_ROOT / "data" / "1_extracted_facts" / "extract_manifest.json"
    embeddings_path = PROJECT_ROOT / "data" / "4_embeddings" / "embeddings.json"
    
    return {
        "cwd": os.getcwd(),
        "project_root": str(PROJECT_ROOT),
        "manifest_exists": manifest_path.exists(),
        "embeddings_exists": embeddings_path.exists(),
        "files_found": results
    }

@app.post("/api/chat", response_model=ChatResponse)
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: QueryRequest):
    try:
        query = request.query
        logger.info(f"Received query: {query}")

        # Phase 1: Query Sanitization & Classification
        p1_result = run_phase_1(query)
        logger.info(f"Phase 1 result: {p1_result}")

        # --- Dynamic URL Addition Feature ---
        if p1_result.get("intent", {}).get("classification") == "URL_ADDITION":
            url = p1_result["sanitized_query"]
            logger.info(f"URL Addition triggered for: {url}")
            
            # 1. Update SourceWebsites.md
            source_file = PROJECT_ROOT / "SourceWebsites.md"
            with open(source_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            if url not in content:
                with open(source_file, "a", encoding="utf-8") as f:
                    f.write(f"\n- {url}")
                logger.info(f"Added {url} to SourceWebsites.md")
                
                # --- NEW: Push to GitHub ---
                github_token = os.environ.get("GITHUB_TOKEN")
                if github_token:
                    try:
                        import subprocess
                        # Configure git
                        subprocess.run(["git", "config", "user.email", "bot@rag-assistant.com"], check=True)
                        subprocess.run(["git", "config", "user.name", "RAG Assistant Bot"], check=True)
                        
                        # Add and commit
                        subprocess.run(["git", "add", "SourceWebsites.md"], check=True)
                        subprocess.run(["git", "commit", "-m", f"Automated: added new fund source {url}"], check=True)
                        
                        # Push using token
                        repo_url = f"https://{github_token}@github.com/mvinoth1979/mutual-fund-rag-assistant.git"
                        subprocess.run(["git", "push", repo_url, "master"], check=True)
                        logger.info("Successfully pushed SourceWebsites.md to GitHub")
                    except Exception as git_err:
                        logger.error(f"Failed to push to GitHub: {git_err}")
                else:
                    logger.warning("GITHUB_TOKEN not found. Changes to SourceWebsites.md will only be local and temporary.")
            else:
                logger.info(f"URL {url} already exists in SourceWebsites.md")

            # 2. Trigger Full Ingestion (Phase 0 to 7)
            # We run this in the background or blocking? User wants to know when it's available.
            # Blocking for now to ensure T6 means it's ready.
            from run_ingestion import run_full_ingestion
            try:
                run_full_ingestion()
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                return ChatResponse(
                    text="New fund source added successfully. I have updated my knowledge base and am now ready to answer questions about this scheme.",
                    source_url=url,
                    footer_date=today,
                    terminal_state="T6"
                )
            except Exception as e:
                logger.error(f"Ingestion failed: {e}")
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                return ChatResponse(
                    text="I detected a URL but failed to ingest the data. Please ensure it's a valid mutual fund page.",
                    source_url=None,
                    footer_date=today,
                    terminal_state="T4"
                )

        if p1_result["blocked_by_pii"] or p1_result["terminal_response"]:
            res = p1_result["terminal_response"]
            return ChatResponse(
                text=res["text"],
                source_url=res["source_url"],
                footer_date=res["footer_date"],
                terminal_state=res["terminal_state"]
            )

        # Phase 2: Corpus Retrieval
        p2_result = run_phase_2(p1_result["sanitized_query"])
        if not p2_result.filtered_candidates:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            return ChatResponse(
                text="I do not have that information in my current sources.",
                source_url=None,
                footer_date=today,
                terminal_state="T3"
            )

        # Phase 3: Context Assembly
        context_string, source_url, doc_id = context_pipeline.execute(p2_result.filtered_candidates)
        if not context_string:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            return ChatResponse(
                text="I do not have that information in my current sources.",
                source_url=None,
                footer_date=today,
                terminal_state="T3"
            )

        # Phase 4: Response Generation
        p4_result = generation_pipeline.generate_response(context_string, p1_result["sanitized_query"])
        
        # Phase 5: Compliance Check
        p5_result = compliance_pipeline.validate(
            raw_response=p4_result.text,
            source_context=context_string,
            source_url=source_url
        )

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return ChatResponse(
            text=p5_result.response,
            source_url=source_url,
            footer_date=today,
            terminal_state=p5_result.status
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return ChatResponse(
            text=f"An internal error occurred: {str(e)}",
            source_url=None,
            footer_date=today,
            terminal_state="ERROR"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
