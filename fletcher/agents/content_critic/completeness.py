import json

from fletcher.llm.client import GenerationConfig, LLMClient
from fletcher.llm.message import Message
from fletcher.agents.schemas import CriticVerdict
from fletcher.rag.lecture_notes.retriever import LectureNoteRetriever


class CompletenessCritic:
    def __init__(self, client: LLMClient, retriever: LectureNoteRetriever):
        self.client = client
        self.retriever = retriever

    def evaluate(
        self,
        explanation: str,
        config: GenerationConfig | None = None,
    ) -> CriticVerdict:
        context_passages = self.retriever.retrieve(explanation)
        context = "\n\n".join(context_passages)

        messages = [
            Message(
                role="system",
                content=(
                    "You are a completeness critic for a computer science course. "
                    "You are given a reference passage from a textbook and a student explanation. "
                    "Evaluate ONLY whether the student explanation is missing key concepts "
                    "that appear in the reference passage. Do not evaluate correctness. "
                    "Respond with ONLY a JSON object in this exact format, no other text:\n"
                    '{"flagged": true or false, "confidence": 0.0 to 1.0, "reasoning": "..."}'
                ),
            ),
            Message(
                role="user",
                content=(
                    f"Reference passage:\n{context}\n\n"
                    f"Student explanation:\n{explanation}"
                ),
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

        return CriticVerdict(role="completeness", **data)