import time

from fletcher.llm.client import GenerationConfig, LLMClient, LLMResponse
from fletcher.llm.message import Message


class FakeLLMClient(LLMClient):
    def __init__(self, canned_response: str = "This is a fake response."):
        self.canned_response = canned_response

    def generate(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        start = time.perf_counter()
        text = self.canned_response
        latency_ms = (time.perf_counter() - start) * 1000

        prompt_tokens = sum(len(m.content.split()) for m in messages)
        completion_tokens = len(text.split())

        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )