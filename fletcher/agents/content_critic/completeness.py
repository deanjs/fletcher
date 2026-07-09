import json

from fletcher.llm.client import GenerationConfig, LLMClient, LLMResponse
from fletcher.llm.message import Message
from fletcher.agents.schemas import CriticVerdict
from fletcher.rag.lecture_notes.retriever import LectureNoteRetriever


class CompletenessCritic:
    def __init__(self, client: LLMClient, retriever: LectureNoteRetriever):
        self.client = client
        self.retriever = retriever
        self.last_response: LLMResponse | None = None
        self.last_debate_response: LLMResponse | None = None
        self.last_debate_text: str = ""

    def evaluate(
        self,
        explanation: str,
        config: GenerationConfig | None = None,
        debate_history: list[dict] | None = None,
    ) -> CriticVerdict:
        context_passages = self.retriever.retrieve(explanation)
        context = "\n\n".join(context_passages)
        debate_context = self._build_debate_context(debate_history)

        messages = [
            Message(
                role="system",
                content=(
                    "You are a completeness critic for a computer science course. "
                    "You are given a reference passage from a textbook and a student explanation. "
                    "Evaluate ONLY whether the student explanation is missing key concepts "
                    "that appear in the reference passage. Do not evaluate correctness. "
                    f"{debate_context}"
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
        self.last_response = response
        return self._parse(response.text)

    def compose_debate_turn(
        self,
        explanation: str,
        verdict: CriticVerdict,
        debate_history: list[dict] | None = None,
        config: GenerationConfig | None = None,
    ) -> str:
        debate_context = self._build_debate_context(debate_history)
        round_instruction = (
            "State your current position to the other critics in two or three sentences. "
            "Defend your judgment and address any disagreement directly."
            if debate_history
            else "State your initial position to the other critics in two or three sentences."
        )
        messages = [
            Message(
                role="system",
                content=(
                    "You are the Completeness Critic in a multi-agent debate. "
                    "Write a short message to the other critics about your current judgment. "
                    "Focus on missing concepts supported by the reference material.\n\n"
                    f"{debate_context}"
                    f"{round_instruction}"
                ),
            ),
            Message(
                role="user",
                content=(
                    f"Student explanation:\n\n{explanation}\n\n"
                    f"Your current verdict:\n"
                    f"- flagged: {verdict.flagged}\n"
                    f"- confidence: {verdict.confidence}\n"
                    f"- reasoning: {verdict.reasoning}"
                ),
            ),
        ]

        response = self.client.generate(messages, config=config)
        self.last_debate_response = response
        self.last_debate_text = response.text
        return response.text

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

    def _build_debate_context(self, debate_history: list[dict] | None) -> str:
        if not debate_history:
            return ""

        previous_round = debate_history[-1]
        prior_verdicts = previous_round.get("verdicts", {})
        lines = []
        for other_role, verdict in prior_verdicts.items():
            if other_role == "completeness":
                continue
            role_name = other_role.replace("_", " ").title()
            lines.append(
                f"In the previous round, the {role_name} Critic set flagged={verdict['flagged']} "
                f"with confidence={verdict['confidence']:.2f} and reasoning: {verdict['reasoning']}"
            )

        if not lines:
            return ""

        return (
            "Reconsider your evaluation in light of the previous round judgments from other critics.\n"
            + "\n".join(lines)
            + "\n\n"
        )
