import json

from fletcher.llm.client import GenerationConfig, LLMClient, LLMResponse
from fletcher.llm.message import Message
from fletcher.agents.schemas import CriticVerdict
from fletcher.rag.lecture_notes.retriever import LectureNoteRetriever


class CompletenessCritic:
    def __init__(self, client: LLMClient, retriever: LectureNoteRetriever | None = None):
        self.client = client
        self.retriever = retriever
        self.last_response: LLMResponse | None = None

    def evaluate(
        self,
        explanation: str,
        config: GenerationConfig | None = None,
        debate_history: list[dict] | None = None,
        request_message: bool = False,
    ) -> CriticVerdict:
        context = self._build_reference_context(explanation)
        debate_context = self._build_debate_context(debate_history)

        if self.retriever is not None:
            instruction = (
                "You are a completeness critic for a computer science course. "
                "You are given a reference passage from a textbook and a student explanation. "
                "Evaluate ONLY whether the student explanation is missing key concepts "
                "that appear in the reference passage. Do not evaluate correctness. "
            )
            user_content = (
                f"Reference passage:\n{context}\n\n"
                f"Student explanation:\n{explanation}"
            )
        else:
            instruction = (
                "You are a completeness critic for a computer science course. "
                "No reference passage is available for this evaluation, so you must judge "
                "completeness using only your own knowledge of the topic. "
                "Evaluate ONLY whether the student explanation is missing key concepts "
                "a correct explanation of this topic would be expected to cover. "
                "Do not evaluate correctness. "
            )
            user_content = f"Student explanation:\n{explanation}"

        if request_message:
            message_instruction = (
                "Also include a \"message_to_others\" field: 2-3 sentences stating your "
                "position to the other critics, defending your judgment and directly "
                "addressing any disagreement from the previous round.\n"
                if debate_history
                else "Also include a \"message_to_others\" field: 2-3 sentences stating your "
                "initial position to the other critics.\n"
            )
            schema_line = (
                '{"flagged": true or false, "confidence": 0.0 to 1.0, "reasoning": "...", '
                '"message_to_others": "..."}'
            )
        else:
            message_instruction = ""
            schema_line = '{"flagged": true or false, "confidence": 0.0 to 1.0, "reasoning": "..."}'

        messages = [
            Message(
                role="system",
                content=(
                    f"{instruction}"
                    f"{debate_context}"
                    f"{message_instruction}"
                    "Respond with ONLY a JSON object in this exact format, no other text:\n"
                    f"{schema_line}"
                ),
            ),
            Message(
                role="user",
                content=user_content,
            ),
        ]

        response = self.client.generate(messages, config=config)
        self.last_response = response
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

    def _build_reference_context(self, explanation: str) -> str:
        if self.retriever is None:
            return ""

        context_passages = self.retriever.retrieve(explanation)
        return "\n\n".join(context_passages)

    def _build_debate_context(self, debate_history: list[dict] | None) -> str:
        if not debate_history:
            return ""

        previous_round = debate_history[-1]
        prior_verdicts = previous_round.get("verdicts", {})

        own_verdict = prior_verdicts.get("completeness")
        others = {k: v for k, v in prior_verdicts.items() if k != "completeness"}
        if not own_verdict and not others:
            return ""

        payload = {}
        if own_verdict:
            payload["your_previous_verdict"] = {
                "flagged": own_verdict["flagged"],
                "confidence": own_verdict["confidence"],
                "reasoning": own_verdict["reasoning"],
            }
        if others:
            payload["other_critics_previous_round"] = {
                other_key: {
                    "flagged": v["flagged"],
                    "confidence": v["confidence"],
                    "message": v.get("message_to_others") or v.get("reasoning", ""),
                }
                for other_key, v in others.items()
            }

        instruction = (
            "Reconsider your evaluation. Keep your own position unless another critic's "
            "argument genuinely changes your assessment — do not switch just because someone "
            "else disagreed with you.\n"
        )
        return instruction + json.dumps(payload) + "\n\n"
