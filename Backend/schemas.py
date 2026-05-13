
from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    question:   str
    top_k:      Optional[int]  = None   # override CANDIDATE_K per-request
    debug:      Optional[bool] = False  # return extracted facts + verification
    session_id: Optional[str]  = None   # conversation session for multi-turn memory


class SourceItem(BaseModel):
    filename: str
    page:     Optional[int]
    score:    Optional[float]


class FactVerificationItem(BaseModel):
    fact:           str
    tag_fname:      str
    tag_page:       Optional[str]
    similarity:     float
    verbatim_match: bool
    matched_preview: Optional[str]


class QueryResponse(BaseModel):
    question:   str
    answer:     str
    sources:    List[SourceItem]
    elapsed_s:  float
    session_id: Optional[str]  = None   # echoed back so client can track the session
    cache_hit:  Optional[bool] = None   # true if answer came from semantic cache
    # optional debug fields
    extracted_facts_preview: Optional[str]                    = None
    verification:            Optional[List[FactVerificationItem]] = None


class TokenRequest(BaseModel):
    api_key: str


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    expires_in:   int
