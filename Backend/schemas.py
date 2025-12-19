
from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None     # optional override for candidate_k
    debug: Optional[bool] = False   # if true, return extractor preview & verification

class SourceItem(BaseModel):
    filename: str
    page: Optional[int]
    score: Optional[float]

class FactVerificationItem(BaseModel):
    fact: str
    tag_fname: str
    tag_page: Optional[str]
    similarity: float
    verbatim_match: bool
    matched_preview: Optional[str]

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceItem]
    elapsed_s: float
    # optional debug fields
    extracted_facts_preview: Optional[str] = None
    verification: Optional[List[FactVerificationItem]] = None
