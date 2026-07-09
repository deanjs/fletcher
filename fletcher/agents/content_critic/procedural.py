import json

from fletcher.llm.client import GenerationConfig, LLMClient
from fletcher.llm.message import Message
from fletcher.agents.schemas import CriticVerdict


class ProceduralCritic:
    def __init__(self, client: LLMClient):
        self.client = client

    def evaluate(self, explanation: str, config: GenerationConfig | None = None) -> CriticVerdict:
        messages = [
            Message(
                role="system",
                content=(
                    "You are a procedural fidelity critic for a computer science course. "
                    "Evaluate ONLY whether the steps and order of applying a concept are "
                    "logically correct. Do not evaluate whether definitions are correct.\n\n"
                    "Respond with ONLY a JSON object in this exact format, no other text:\n"
                    '{"flagged": true or false, "confidence": 0.0 to 1.0, "reasoning": "..."}'
                ),
            ),
            Message(
                role="user",
                content=f"Student explanation:\n\n{explanation}",
            ),
        ]

        response = self.client.generate(messages, config=config)
        return self._parse(response.text)

    def _parse(self, text: str) -> CriticVerdict:
        try:
            data = json.loads(text.strip())
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1:
                raise ValueError(f"Could not find JSON object in critic output:\n{text}")
            data = json.loads(text[start:end + 1])

        return CriticVerdict(role="procedural", **data)