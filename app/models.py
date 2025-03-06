from pydantic import BaseModel
from typing import List, Optional

class IdealCandidateRequest(BaseModel):
    languages: Optional[List[str]] = None
    min_commits: Optional[int] = None
    min_stars: Optional[int] = None
    # More criteria fields if desired

class CandidateResponse(BaseModel):
    username: str
    similarity_score: float
    hire_decision: str
    # Additional metrics if desired