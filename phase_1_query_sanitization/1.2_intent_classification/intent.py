import re
from typing import List, Literal, Optional
from pydantic import BaseModel

class IntentResult(BaseModel):
    classification: Literal["FACTUAL", "ADVISORY", "URL_ADDITION", "UNCLEAR"]
    advisory_score: float
    triggers_matched: List[str] = []
    detected_url: Optional[str] = None

class IntentClassifier:
    # Captures full URL including path and query params
    URL_PATTERN = r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2})|[/?:@!$&'()*+,;=._~-])+"
    
    ADVISORY_TRIGGERS = [
        r"\bshould\b",
        r"\bbetter\b",
        r"\bbest\b",
        r"\brecommend\b",
        r"\bsafe\b",
        r"\boutperform\b",
        r"invest now",
        r"\bbuy\b",
        r"\bsell\b",
    ]

    FACTUAL_TRIGGERS = [
        r"expense ratio",
        r"exit load",
        r"minimum sip",
        r"min sip",
        r"minimum lumpsum",
        r"min lumpsum",
        r"benchmark",
        r"riskometer",
        r"risk level",
        r"nav",
        r"aum",
        r"asset under management",
        r"fund manager",
        r"who manages",
        r"\bmanager\b",
        r"inception date",
        r"launch date",
        r"category",
        r"tax",
        r"taxation",
        r"holdings",
        r"portfolio",
        r"stocks",
        r"sector",
        r"performance",
        r"returns",
        r"investment objective",
        r"objective",
        r"fund size",
        r"total aum",
        r"who",
        r"tell me about",
        r"information on",
    ]

    def classify(self, query: str) -> IntentResult:
        query_lower = query.lower()

        # 1. URL Detection (Priority)
        url_match = re.search(self.URL_PATTERN, query_lower)
        if url_match:
            return IntentResult(
                classification="URL_ADDITION",
                advisory_score=0.0,
                triggers_matched=["URL_DETECTED"],
                detected_url=url_match.group(0)
            )

        advisory_hits = 0
        matched_triggers: List[str] = []

        for pattern in self.ADVISORY_TRIGGERS:
            if re.search(pattern, query_lower):
                advisory_hits += 1
                matched_triggers.append(f"ADV:{pattern}")

        factual_hits = 0
        for pattern in self.FACTUAL_TRIGGERS:
            if re.search(pattern, query_lower):
                factual_hits += 1
                matched_triggers.append(f"FACT:{pattern}")

        total_hits = advisory_hits + factual_hits

        if total_hits == 0:
            return IntentResult(
                classification="UNCLEAR",
                advisory_score=0.0,
                triggers_matched=[],
            )

        advisory_score = advisory_hits / total_hits

        if advisory_score >= 0.5:
            classification = "ADVISORY"
        else:
            classification = "FACTUAL"

        return IntentResult(
            classification=classification,
            advisory_score=round(advisory_score, 3),
            triggers_matched=matched_triggers,
        )

def run_intent_gate(query: str) -> IntentResult:
    classifier = IntentClassifier()
    return classifier.classify(query)
