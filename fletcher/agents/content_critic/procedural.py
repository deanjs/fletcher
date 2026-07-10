import json
from typing import TYPE_CHECKING

from fletcher.llm.client import GenerationConfig, LLMClient, LLMResponse
from fletcher.llm.message import Message
from fletcher.agents.schemas import CriticVerdict

if TYPE_CHECKING:
    from fletcher.rag.lecture_notes.retriever import LectureNoteRetriever

PERSONA_PROMPTS = {
    "strict": (
        "Be highly critical. Flag any potential procedural issues, even minor ones. "
        "When in doubt, flag it."
    ),
    "merciful": (
        "Only flag clear and significant procedural errors. "
        "Ignore minor imprecisions or incomplete but not incorrect steps."
    ),
    "neutral": "",
}


class ProceduralCritic:
    def __init__(
        self,
        client: LLMClient,
        persona: str = "neutral",
        retriever: "LectureNoteRetriever | None" = None,
    ):
        self.client = client
        self.persona = persona
        self.retriever = retriever
        self.last_response: LLMResponse | None = None

    def evaluate(
        self,
        explanation: str,
        config: GenerationConfig | None = None,
        debate_history: list[dict] | None = None,
        debate_key: str | None = None,
        request_message: bool = False,
    ) -> CriticVerdict:
        persona_instruction = PERSONA_PROMPTS.get(self.persona, "")
        persona_line = f" {persona_instruction}" if persona_instruction else ""
        rag_context = self._build_rag_context(explanation)
        debate_context = self._build_debate_context(debate_history, role=debate_key or "procedural")

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
                    "You are a procedural fidelity critic for a computer science course. "
                    "Evaluate ONLY whether the steps and order of applying a concept are "
                    "logically correct. Do not evaluate whether definitions are correct."
                    f"{persona_line}\n\n"
                    f"{rag_context}"
                    f"{debate_context}"
                    f"{message_instruction}"
                    "Respond with ONLY a JSON object in this exact format, no other text:\n"
                    f"{schema_line}"
                ),
            ),
            Message(
                role="user",
                content=f"Student explanation:\n\n{explanation}",
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

        return CriticVerdict(role="procedural", **data)

    def _build_rag_context(self, explanation: str) -> str:
        if self.retriever is None:
            return ""

        context_passages = self.retriever.retrieve(explanation)
        if not context_passages:
            return ""

        context = "\n\n".join(context_passages)
        return (
            "Use the following retrieved reference passages as grounding evidence when available.\n"
            f"Reference passages:\n{context}\n\n"
        )

    def _build_debate_context(self, debate_history: list[dict] | None, role: str) -> str:
        if not debate_history:
            return ""

        previous_round = debate_history[-1]
        prior_verdicts = previous_round.get("verdicts", {})

        own_verdict = prior_verdicts.get(role)
        others = {k: v for k, v in prior_verdicts.items() if k != role}
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
