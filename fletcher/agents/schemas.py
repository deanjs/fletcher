from typing import Literal
from pydantic import BaseModel


class CriticVerdict(BaseModel):
    role: Literal["conceptual", "procedural", "completeness"]
    flagged: bool
    reasoning: str
    confidence: float