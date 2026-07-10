import time

from fletcher.llm.client import GenerationConfig, LLMClient, LLMResponse
from fletcher.llm.message import Message


class FakeLLMClient(LLMClient):
    def __init__(self, canned_response: str | None = None):
        self.canned_response = canned_response

    def generate(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        start = time.perf_counter()
        text = self.canned_response or self._default_response(messages)
        latency_ms = (time.perf_counter() - start) * 1000

        prompt_tokens = sum(len(m.content.split()) for m in messages)
        completion_tokens = len(text.split())

        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )

    def _default_response(self, messages: list[Message]) -> str:
        prompt = "\n".join(message.content for message in messages)
        prompt_lower = prompt.lower()
        flagged = self._should_flag(prompt_lower)

        if "respond with only a json object" in prompt_lower:
            if '"confidence"' in prompt_lower and '"notes"' in prompt_lower:
                return '{"confidence": 0.95, "notes": "The critique is sufficient for this fake smoke run."}'

            role = "conceptual"
            if "procedural fidelity critic" in prompt_lower:
                role = "procedural"
            elif "completeness critic" in prompt_lower:
                role = "completeness"

            reasoning = (
                "The explanation contains the known hard-negative misconception."
                if flagged
                else "The explanation matches the expected smoke-test pattern."
            )
            if "message_to_others" in prompt_lower:
                return (
                    f'{{"flagged": {str(flagged).lower()}, "confidence": 0.9, '
                    f'"reasoning": "{reasoning}", '
                    f'"message_to_others": "I would {"flag" if flagged else "not flag"} this case."}}'
                )
            return (
                f'{{"flagged": {str(flagged).lower()}, "confidence": 0.9, '
                f'"reasoning": "{reasoning}"}}'
            )

        if "verdict: flagged" in prompt_lower or "verdict: ok" in prompt_lower:
            critique = (
                "The explanation should be flagged because it contains the known misconception."
                if flagged
                else "The explanation looks accurate for this fake smoke run."
            )
            verdict = "FLAGGED" if flagged else "OK"
            return f"{critique}\n\nVERDICT: {verdict}"

        return (
            "The explanation should be flagged because it contains the known misconception."
            if flagged
            else "The explanation looks accurate for this fake smoke run."
        )

    def _should_flag(self, prompt_lower: str) -> bool:
        target_text = prompt_lower
        if "student explanation:" in prompt_lower:
            target_text = prompt_lower.rsplit("student explanation:", 1)[-1]

        hard_negative_markers = [
            "only at the token immediately before",
            "never need collision handling",
            "do not need collision handling",
            "does not need collision handling",
            "unique index",
        ]
        return any(marker in target_text for marker in hard_negative_markers)
