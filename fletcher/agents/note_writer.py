from fletcher.llm.client import GenerationConfig, LLMClient
from fletcher.llm.message import Message


class NoteWriter:
    def __init__(self, client: LLMClient):
        self.client = client

    def rewrite(
        self,
        critique_text: str,
        tone_sample: str,
        config: GenerationConfig | None = None,
    ) -> str:
        messages = [
            Message(
                role="system",
                content=(
                    "You rewrite technical feedback into a short review note. "
                    "Match the tone, vocabulary level, and sentence style shown in the "
                    "student's writing sample below, while preserving the full content "
                    "and accuracy of the original feedback. Do not soften or omit any "
                    "flagged issues."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"Student's writing sample (match this tone):\n{tone_sample}\n\n"
                    f"Original feedback to rewrite:\n{critique_text}"
                ),
            ),
        ]

        response = self.client.generate(messages, config=config)
        return response.text