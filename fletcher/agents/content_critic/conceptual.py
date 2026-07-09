import json

from fletcher.llm.client import GenerationConfig, LLMClient
from fletcher.llm.message import Message
from fletcher.agents.schemas import CriticVerdict

PERSONA_PROMPTS = {
    "strict": (
        "Be highly critical. Flag any potential conceptual issues, even minor ones. "
        "When in doubt, flag it."
    ),
    "merciful": (
        "Only flag clear and significant conceptual errors. "
        "Ignore minor imprecisions or incomplete but not incorrect statements."
    ),
    "neutral": "",
}


class ConceptualCritic:
    def __init__(self, client: LLMClient, persona: str = "neutral"):
        self.client = client
        self.persona = persona

    def evaluate(self, explanation: str, config: GenerationConfig | None = None) -> CriticVerdict:
        persona_instruction = PERSONA_PROMPTS.get(self.persona, "")
        persona_line = f" {persona_instruction}" if persona_instruction else ""

        messages = [
            Message(
                role="system",
                content=(
                    "You are a conceptual fidelity critic for a computer science course. "
                    "Evaluate ONLY whether the concepts and definitions in the student's "
                    "explanation are correct. Do not evaluate procedural steps or ordering."
                    f"{persona_line}\n\n"
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
                raise ValueError(f"Could not find JSON in critic output:\n{text}")
            data = json.loads(text[start:end + 1])

        return CriticVerdict(role="conceptual", **data)