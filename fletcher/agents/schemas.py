from typing import Literal
from pydantic import BaseModel


class CriticVerdict(BaseModel):
    role: Literal["conceptual", "procedural", "completeness"]
    flagged: bool
    reasoning: str
    confidence: float
    # Only populated when evaluate() is called with request_message=True
    # (Stage-1 / debate contexts). Empty for Stage-2 (R-axis), which never
    # debates and so never needs a message addressed to other critics.
    message_to_others: str = ""