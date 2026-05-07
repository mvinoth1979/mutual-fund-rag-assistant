import re
import json
from pathlib import Path
from typing import Optional
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class ResolvedQuery(BaseModel):
    original_query: str
    normalized_query: str
    mentioned_funds: list[str]
    fund_resolution_confidence: float
    fact_type: Optional[str]
    fact_confidence: float
    is_ambiguous: bool
    advisory_trigger: bool = False
    advisory_reason: str = ""

class EntityResolver:
    def __init__(self):
        self.FUND_MAP = {}
        self.DOC_ID_TO_URL = {}
        self._load_entities()

    def _load_entities(self):
        manifest_path = PROJECT_ROOT / "data" / "1_extracted_facts" / "extract_manifest.json"
        if not manifest_path.exists():
            return

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            
            for item in manifest.get("results", []):
                doc_id = item["doc_id"]
                url = item["source_url"]
                self.DOC_ID_TO_URL[doc_id] = url
                
                # Derive name from URL slug
                slug = url.split("/")[-1]
                # Remove common suffixes and handle special characters
                clean_name = slug.replace("-direct-growth", "").replace("-direct", "").replace("-growth", "").replace("-", " ")
                # Remove parentheses for better matching
                clean_name = clean_name.replace("(", "").replace(")", "")
                # Normalize spaces
                clean_name = " ".join(clean_name.split())
                
                # Add various aliases
                self.FUND_MAP[clean_name.lower()] = doc_id
                
                # Add shorthands
                if "the wealth company" in clean_name.lower():
                    shorthand = clean_name.lower().replace("the wealth company ", "")
                    self.FUND_MAP[shorthand] = doc_id
                
                if "icici prudential" in clean_name.lower():
                    shorthand = clean_name.lower().replace("icici prudential ", "")
                    self.FUND_MAP[shorthand] = doc_id
                    
                    # More specific shorthands
                    if "fund" in shorthand:
                        shorthand2 = shorthand.replace(" fund", "")
                        self.FUND_MAP[shorthand2] = doc_id

        except Exception as e:
            print(f"Error loading entities: {e}")

    FACT_MAP = {
        "expense ratio": "expense_ratio",
        "charges": "expense_ratio",
        "fees": "expense_ratio",
        "cost": "expense_ratio",
        "exit load": "exit_load",
        "redemption fee": "exit_load",
        "exit charge": "exit_load",
        "min sip": "min_sip",
        "minimum sip": "min_sip",
        "systematic investment plan": "min_sip",
        "sip amount": "min_sip",
        "min lumpsum": "min_lumpsum",
        "minimum lumpsum": "min_lumpsum",
        "lumpsum amount": "min_lumpsum",
        "benchmark": "benchmark",
        "nav": "nav",
        "net asset value": "nav",
        "aum": "aum",
        "assets under management": "aum",
        "asset under management": "aum",
        "fund size": "aum",
        "riskometer": "risk_level",
        "risk level": "risk_level",
        "risk": "risk_level",
        "fund manager": "fund_manager",
        "manager": "fund_manager",
        "inception date": "inception_date",
        "launch date": "inception_date",
        "category": "category",
        "tax": "tax",
        "taxation": "tax",
        "holdings": "holdings",
        "portfolio": "holdings",
        "investment objective": "investment_objective",
        "objective": "investment_objective",
        "overview": "overview",
        "total aum": "total_aum",
    }

    COMPARISON_WORDS = [
        "better", "worse", "superior", "inferior", "compare",
        "comparison", "versus", "vs", "difference", "diff",
    ]

    def resolve(self, original_query: str, normalized_query: str) -> ResolvedQuery:
        text = normalized_query.lower()
        mentioned_funds: list[str] = []
        
        # Sort aliases by length descending to prioritize more specific matches (e.g., 'Axis Large Cap Fund' over 'Large Cap Fund')
        sorted_aliases = sorted(self.FUND_MAP.keys(), key=len, reverse=True)
        
        # Track which parts of the text have already been matched to avoid double-counting or partial matches
        matched_indices = []

        for alias in sorted_aliases:
            if alias in text:
                # Check if this alias is already covered by a longer match
                start_idx = text.find(alias)
                end_idx = start_idx + len(alias)
                
                is_submatch = False
                for m_start, m_end in matched_indices:
                    if start_idx >= m_start and end_idx <= m_end:
                        is_submatch = True
                        break
                
                if not is_submatch:
                    doc_id = self.FUND_MAP[alias]
                    if doc_id not in mentioned_funds:
                        mentioned_funds.append(doc_id)
                        matched_indices.append((start_idx, end_idx))

        if len(mentioned_funds) == 1:
            fund_confidence = 1.0
        elif len(mentioned_funds) > 1:
            fund_confidence = 1.0
        else:
            fund_confidence = 0.0

        fact_type: Optional[str] = None
        fact_confidence = 0.0
        
        # Sort keys by length descending to match longest phrase first
        sorted_keys = sorted(self.FACT_MAP.keys(), key=len, reverse=True)
        for alias in sorted_keys:
            if alias in text:
                fact_type = self.FACT_MAP[alias]
                fact_confidence = 1.0
                break

        has_comparison = any(word in text for word in self.COMPARISON_WORDS)
        is_ambiguous = len(mentioned_funds) > 1 and has_comparison

        advisory_trigger = is_ambiguous
        advisory_reason = ""
        if is_ambiguous:
            advisory_reason = (
                "Multiple fund mentions with explicit comparison detected. "
                "Advisory refusal triggered per architecture enforcement."
            )

        return ResolvedQuery(
            original_query=original_query,
            normalized_query=normalized_query,
            mentioned_funds=mentioned_funds,
            fund_resolution_confidence=fund_confidence,
            fact_type=fact_type,
            fact_confidence=fact_confidence,
            is_ambiguous=is_ambiguous,
            advisory_trigger=advisory_trigger,
            advisory_reason=advisory_reason,
        )

def run_entity_resolver(original_query: str, normalized_query: str) -> ResolvedQuery:
    return EntityResolver().resolve(original_query, normalized_query)
