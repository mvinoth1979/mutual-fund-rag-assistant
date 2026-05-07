"""
Phase 2.1: Query Normalization
===============================
Transforms raw user query into canonical form.
Logic: Lowercase -> punctuation strip -> abbreviation expansion -> synonym mapping -> fund alias canonicalization.
Enforcement: Fixed dictionary; no learned components.
"""

import re
from pydantic import BaseModel


class NormalizedQuery(BaseModel):
    original: str
    normalized: str
    transformations: list[str]


class QueryNormalizer:
    """
    Rule-based query normalizer using fuzzy logic dictionary.
    """
    
    # Load fuzzy mapping from external file
    try:
        from .fuzzy_dictionary import FUZZY_MAPPING
    except ImportError:
        from fuzzy_dictionary import FUZZY_MAPPING
    
    @classmethod
    def _get_sorted_patterns(cls) -> list[tuple[str, str]]:
        """Sort patterns by length descending to ensure greedy matching."""
        return sorted(cls.FUZZY_MAPPING.items(), key=lambda x: len(x[0]), reverse=True)

    def normalize(self, query: str) -> NormalizedQuery:
        original = query.strip()
        transformations: list[str] = []

        # 1. Lowercase
        text = original.lower()

        # 2. Basic cleanup (punctuation to spaces)
        # Keep dots for some acronyms initially? No, architecture says strip punctuation
        text = re.sub(r"[^\w\s]", " ", text)

        # 3. Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()

        # 4. Multi-pass normalization (to handle phrases and then components)
        patterns = self._get_sorted_patterns()
        
        # We do two passes: 
        # Pass 1: Exact phrase replacements (longest first)
        for phrase, canonical in patterns:
            # Word boundary check for non-phrase patterns
            if " " in phrase:
                # Phrase replacement
                if phrase in text:
                    text = text.replace(phrase, canonical)
                    transformations.append(f"Phrase match: {phrase} -> {canonical}")
            else:
                # Word-by-word replacement with regex
                new_text = re.sub(rf"\b{re.escape(phrase)}\b", canonical, text)
                if new_text != text:
                    text = new_text
                    transformations.append(f"Word match: {phrase} -> {canonical}")

        # Final cleanup to remove redundant words (e.g. "small cap fund fund")
        text = re.sub(r"\b(fund|scheme)\s+\1\b", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()

        return NormalizedQuery(
            original=original,
            normalized=text,
            transformations=transformations,
        )


def run_normalizer(query: str) -> NormalizedQuery:
    """Convenience entry-point."""
    return QueryNormalizer().normalize(query)
