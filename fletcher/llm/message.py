from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]

@dataclass(frozen=True)
class Message:
    role: Role
    content: str