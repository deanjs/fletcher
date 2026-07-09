from fletcher.llm.client import GenerationConfig, LLMClient
from fletcher.llm.message import Message


class SelfCritique:
    def __init__(self, client: LLMClient):
        self.client = client

    def critique(self, explanation: str, config: GenerationConfig | None = None) -> str:
        messages = [
            Message(
                role="system",
                content=(
                    "You are a content fidelity critic for a computer science course. "
                    "Evaluate the student's explanation below for factual and conceptual "
                    "accuracy. Point out any misconceptions or missing key ideas. "
                    "Do not evaluate English proficiency or writing style."
                ),
            ),
            Message(
                role="user",
                content=f"Student explanation:\n\n{explanation}",
            ),
        ]

        response = self.client.generate(messages, config=config)
        return response.text