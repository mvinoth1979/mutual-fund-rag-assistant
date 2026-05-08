"""
Phase 0.5: Embed & Index
========================

Production-grade embedding and indexing pipeline.
Consumes semantic chunks from Phase 0.4 and produces:
- Vector index (Chroma DB persistent) with L2-normalized BGE embeddings
- Structured fact store (SQLite) for direct KV lookups
- Embedding artifacts for audit

Architecture reference: Section 4, Phase 0.5
Enforcement: Dimension 1024; no NaN vectors; metadata binding mandatory
"""

import json
import logging
import math
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# =============================================================================
# Configuration
# =============================================================================

DATA_CHUNKS = Path(os.getenv("DATA_CHUNKS", "./data/3_chunks"))
DATA_NORMALIZED = Path(os.getenv("DATA_NORMALIZED", "./data/2_normalized_text"))
DATA_EMBEDDINGS = Path(os.getenv("DATA_EMBEDDINGS", "./data/4_embeddings"))
DATA_STRUCTURED = Path(os.getenv("DATA_STRUCTURED", "./data/5_structured_facts"))
DATA_CHROMA = Path(os.getenv("DATA_CHROMA", "./data/6_chroma_index"))

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"
EXPECTED_DIM = 1024
GEMINI_DIM = 1024  # Force 1024 to match BGE-large for hybrid compatibility
BATCH_SIZE = 16

# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("phase_0_5_embed")


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    source_url: str
    chunk_type: str
    text: str
    token_count: int
    overlap: bool


@dataclass
class EmbedResult:
    chunk_id: str
    doc_id: str
    source_url: str
    chunk_type: str
    text: str
    embedding: List[float]
    l2_norm: float


@dataclass
class EmbedManifestEntry:
    doc_id: str
    chunks_embedded: int
    facts_stored: int
    error: Optional[str] = None


# =============================================================================
# Embedding Engine
# =============================================================================


class GeminiEmbedder:
    def __init__(self, model_name: str = GEMINI_EMBEDDING_MODEL):
        import google.generativeai as genai
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY must be set for GeminiEmbedder")
        genai.configure(api_key=api_key)
        self.genai = genai
        
        self.model_name = model_name
        
        # Auto-detect available models
        try:
            available_models = [m.name for m in self.genai.list_models() if 'embedContent' in m.supported_generation_methods]
            if available_models:
                # Prioritize text-embedding-004 or the requested model if available
                if model_name in available_models:
                    self.model_name = model_name
                elif "models/text-embedding-004" in available_models:
                    self.model_name = "models/text-embedding-004"
                else:
                    self.model_name = available_models[0]
                logger.info(f"Gemini auto-detected working model: {self.model_name}")
            else:
                logger.warning("No models supporting 'embedContent' found. Falling back to default list.")
        except Exception as e:
            logger.warning(f"Failed to list Gemini models: {e}. Using default candidate list.")

    def embed(self, texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
        if not texts:
            return []
        
        # Models found in your Railway logs
        MODEL_CANDIDATES = [self.model_name] if self.model_name else []
        MODEL_CANDIDATES.extend([
            "models/text-embedding-004",
            "models/gemini-embedding-001",
            "text-embedding-004",
            "gemini-embedding-001",
            "models/embedding-001",
            "embedding-001"
        ])
        # Deduplicate while preserving order
        MODEL_CANDIDATES = list(dict.fromkeys(MODEL_CANDIDATES))
        
        results = []
        batch_size = 30  # Increased batch size to reduce RPM (Requests Per Minute) usage
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            success = False
            last_err = None
            
            import time
            max_retries = 5
            for attempt in range(max_retries):
                success = False
                last_err = None
                
                for model_name in MODEL_CANDIDATES:
                    try:
                        # Force 1024 dimensions for compatibility with BGE-indexed data
                        kwargs = {"model": model_name, "content": batch, "task_type": task_type}
                        if "004" in model_name:
                            kwargs["output_dimensionality"] = 1024
                            
                        response = self.genai.embed_content(**kwargs)
                        results.extend(response["embedding"])
                        success = True
                        self.model_name = model_name 
                        break
                    except Exception as e:
                        last_err = e
                        if "429" in str(e) or "ResourceExhausted" in str(e):
                            # This is a rate limit, don't try other models, just retry later
                            break
                        logger.warning(f"Candidate {model_name} failed: {str(e)}")
                        continue
                
                if success:
                    break
                
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 5 # Exponential backoff: 5s, 10s, 20s...
                    logger.info(f"Rate limited or error. Retrying batch in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    # DIAGNOSTIC: List available models to logs
                    try:
                        available_models = [m.name for m in self.genai.list_models() if 'embedContent' in m.supported_generation_methods]
                        logger.error(f"Embedding failed after retries. Available models: {available_models}")
                    except:
                        pass
                    raise last_err if last_err else ValueError("Batch embedding failed.")
            
            time.sleep(5) # Steady delay between batches to stay under free-tier RPM limits
        return results

class BGEEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        
        # Hard-switch to Gemini as HF is returning 404 for this model
        self.use_gemini = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")) is not None
        if self.use_gemini:
            self.gemini = GeminiEmbedder()
            self.dim = GEMINI_DIM
            logger.info("Using Gemini for embeddings (HF disabled due to 404 errors).")
        else:
            # Local fallback (only for development)
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading local embedding model: {model_name}")
                self.model = SentenceTransformer(model_name)
                self.dim = self.model.get_embedding_dimension()
                self.use_gemini = False
            except ImportError:
                raise ValueError("GEMINI_API_KEY must be set for production embeddings.")

    def embed(self, texts: List[str], task_type: str = "retrieval_query") -> List[List[float]]:
        if not texts:
            return []
        
        if self.use_gemini:
            return self.gemini.embed(texts, task_type=task_type)
        
        # Local fallback
        if self.model:
            embeddings = self.model.encode(
                texts, convert_to_numpy=True, show_progress_bar=False
            )
            return [emb.tolist() for emb in embeddings]
        
        raise ValueError("No embedding backend available. Check GEMINI_API_KEY.")

    @staticmethod
    def l2_normalize(embedding: List[float]) -> List[float]:
        """L2-normalize an embedding vector."""
        arr = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm == 0:
            return arr.tolist()
        return (arr / norm).tolist()


# =============================================================================
# Validation
# =============================================================================


class EmbeddingValidator:
    @staticmethod
    def validate(embedding: List[float], chunk_id: str, expected_dim: int) -> Optional[str]:
        """Return error string if invalid, else None."""
        if len(embedding) != expected_dim:
            return f"Dimension mismatch: expected {expected_dim}, got {len(embedding)}"

        arr = np.array(embedding, dtype=np.float32)
        if np.isnan(arr).any():
            return "NaN values detected"

        if np.isinf(arr).any():
            return "Inf values detected"

        norm = np.linalg.norm(arr)
        if norm == 0:
            return "Zero vector detected"

        # After L2 normalization, norm should be ~1.0
        if abs(norm - 1.0) > 0.01:
            return f"L2 norm not close to 1.0 after normalization: {norm}"

        return None


# =============================================================================
# Storage Backends
# =============================================================================


class ChromaStore:
    def __init__(self, persist_dir: Path):
        try:
            import chromadb
            self.available = True
        except ImportError:
            self.available = False
            logger.warning("chromadb not installed. Skipping Chroma indexing (production uses SimpleVectorStore).")
            return

        self.persist_dir = Path(persist_dir)
        try:
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
            self.collection = self.client.get_or_create_collection("mutual_fund_chunks")
            logger.info(f"Chroma initialized at {self.persist_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize Chroma: {e}")
            self.available = False
            self.collection = None

    def upsert(self, records: List[EmbedResult]) -> int:
        """Upsert embedding records into Chroma."""
        if not self.available or self.collection is None:
            logger.warning("Chroma not available, skipping upsert.")
            return 0
            
        if not records:
            return 0

        ids = [r.chunk_id for r in records]
        embeddings = [r.embedding for r in records]
        documents = [r.text for r in records]
        metadatas = [
            {
                "doc_id": r.doc_id,
                "source_url": r.source_url,
                "chunk_type": r.chunk_type,
            }
            for r in records
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return len(records)

    def count(self) -> int:
        return self.collection.count()


class SQLiteStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS structured_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    fact_type TEXT NOT NULL,
                    value TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    indexed_at TEXT NOT NULL
                )
                """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_facts_doc_type
                ON structured_facts(doc_id, fact_type)
                """)
            conn.commit()
        logger.info(f"SQLite schema ready at {self.db_path}")

    def clear_doc_facts(self, doc_id: str):
        """Remove existing facts for a doc_id to ensure idempotency."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM structured_facts WHERE doc_id = ?",
                (doc_id,),
            )
            conn.commit()

    def load_typed_facts(self, typed_facts_path: Path) -> int:
        """Load typed facts from normalized JSON into SQLite."""
        with open(typed_facts_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        facts = data.get("typed_facts", [])
        doc_id = data.get("doc_id", "")
        source_url = data.get("source_url", "")
        indexed_at = datetime.now(timezone.utc).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            for fact in facts:
                conn.execute(
                    """
                    INSERT INTO structured_facts
                    (doc_id, source_url, fact_type, value, confidence, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc_id,
                        source_url,
                        fact.get("fact_type", ""),
                        fact.get("value", ""),
                        fact.get("confidence", ""),
                        indexed_at,
                    ),
                )
            conn.commit()

        return len(facts)

    def get_fact(self, doc_id: str, fact_type: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT value FROM structured_facts WHERE doc_id = ? AND fact_type = ?",
                (doc_id, fact_type),
            ).fetchone()
            return row[0] if row else None


# =============================================================================
# Persistence
# =============================================================================


def save_embeddings(records: List[EmbedResult], output_dir: Path) -> Path:
    """Save embeddings as JSON for audit/reuse."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "embeddings.json"
    payload = []
    for r in records:
        payload.append(
            {
                "chunk_id": r.chunk_id,
                "doc_id": r.doc_id,
                "source_url": r.source_url,
                "chunk_type": r.chunk_type,
                "text": r.text,
                "embedding": r.embedding,
                "l2_norm": r.l2_norm,
            }
        )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(records)} embeddings to {path}")
    return path


# =============================================================================
# Main Entry Point
# =============================================================================


def run_embed_phase() -> Dict[str, Any]:
    """
    Execute Phase 0.5 Embed & Index over all chunked documents.
    """
    logger.info("=" * 60)
    logger.info("PHASE 0.5: EMBED & INDEX")
    logger.info("=" * 60)

    embedder = BGEEmbedder()
    validator = EmbeddingValidator()
    chroma = ChromaStore(DATA_CHROMA)
    sqlite = SQLiteStore(DATA_STRUCTURED / "facts.db")

    manifest = {
        "run_id": datetime.now(timezone.utc).isoformat(),
        "model": EMBEDDING_MODEL,
        "expected_dim": EXPECTED_DIM,
        "total_chunks": 0,
        "embedded": 0,
        "rejected": 0,
        "facts_stored": 0,
        "results": [],
    }

    all_records: List[EmbedResult] = []

    chunk_files = sorted(DATA_CHUNKS.glob("DOC-*_chunks.jsonl"))
    if not chunk_files:
        logger.warning(f"No chunk files found in {DATA_CHUNKS}")
        return manifest

    # Process each document
    for chunk_file in chunk_files:
        doc_id = chunk_file.stem.replace("_chunks", "")
        logger.info(f"Processing {doc_id} ...")

        entry = EmbedManifestEntry(doc_id=doc_id, chunks_embedded=0, facts_stored=0)

        # 1. Load chunks
        chunks: List[ChunkRecord] = []
        with open(chunk_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    chunks.append(
                        ChunkRecord(
                            chunk_id=obj["chunk_id"],
                            doc_id=obj["doc_id"],
                            source_url=obj["source_url"],
                            chunk_type=obj["chunk_type"],
                            text=obj["text"],
                            token_count=obj["token_count"],
                            overlap=obj.get("overlap", False),
                        )
                    )

        manifest["total_chunks"] += len(chunks)

        # 2. Batch embed
        texts = [c.text for c in chunks]
        raw_embeddings = embedder.embed(texts, task_type="retrieval_document")
        time.sleep(2) # Rate limiting between docs

        doc_records: List[EmbedResult] = []
        rejected = 0
        for chunk, emb in zip(chunks, raw_embeddings):
            if len(emb) != embedder.dim:
                logger.error(f"Validation failed for {chunk.chunk_id}: Dimension mismatch: expected {embedder.dim}, got {len(emb)}")
                rejected += 1
                manifest["rejected"] += 1
                continue

            normalized = embedder.l2_normalize(emb)
            error = validator.validate(normalized, chunk.chunk_id, embedder.dim)
            if error:
                logger.error(f"Validation failed for {chunk.chunk_id}: {error}")
                rejected += 1
                manifest["rejected"] += 1
                continue

            norm = float(np.linalg.norm(np.array(normalized, dtype=np.float32)))
            doc_records.append(
                EmbedResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source_url=chunk.source_url,
                    chunk_type=chunk.chunk_type,
                    text=chunk.text,
                    embedding=normalized,
                    l2_norm=norm,
                )
            )

        # 3. Upsert to Chroma
        if doc_records:
            chroma.upsert(doc_records)
            all_records.extend(doc_records)

        entry.chunks_embedded = len(doc_records)
        manifest["embedded"] += len(doc_records)

        # 4. Load structured facts into SQLite
        facts_file = DATA_NORMALIZED / f"{doc_id}_typed_facts.json"
        if facts_file.exists():
            sqlite.clear_doc_facts(doc_id)
            fact_count = sqlite.load_typed_facts(facts_file)
            entry.facts_stored = fact_count
            manifest["facts_stored"] += fact_count
        else:
            logger.warning(f"Typed facts file not found: {facts_file}")

        if rejected > 0:
            entry.error = f"{rejected} chunks rejected"

        manifest["results"].append(
            {
                "doc_id": entry.doc_id,
                "chunks_embedded": entry.chunks_embedded,
                "facts_stored": entry.facts_stored,
                "error": entry.error,
            }
        )

    # 5. Save embedding artifacts
    if all_records:
        save_embeddings(all_records, DATA_EMBEDDINGS)

    # 6. Final validation
    chroma_count = chroma.count() if chroma.collection else 0
    logger.info(f"Chroma collection count: {chroma_count}")
    manifest["chroma_count"] = chroma_count

    # Save manifest
    manifest_path = DATA_EMBEDDINGS / "embed_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest saved to {manifest_path}")

    logger.info("-" * 60)
    logger.info(
        f"Embed complete: {manifest['embedded']} embedded, "
        f"{manifest['rejected']} rejected, "
        f"{manifest['facts_stored']} facts stored"
    )
    logger.info("=" * 60)

    return manifest


if __name__ == "__main__":
    run_embed_phase()
