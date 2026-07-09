from abc import ABC, abstractmethod
from dataclasses import dataclass

from fletcher.llm.message import Message


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.7
    max_new_tokens: int = 512
    top_p: float = 0.9


@dataclass(frozen=True)
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class LLMClient(ABC):
    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        ...