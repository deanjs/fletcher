import json

from fletcher.llm.client import GenerationConfig, LLMClient, LLMResponse
from fletcher.llm.message import Message

MAX_ITERATIONS = 3


class SelfCritique:
    def __init__(self, client: LLMClient, max_iterations: int = MAX_ITERATIONS):
        self.client = client
        self.max_iterations = max_iterations
        self.last_response: LLMResponse | None = None
        self.last_llm_calls: int = 0
        self.last_iterations_used: int = 0

    def critique(self, explanation: str, config: GenerationConfig | None = None) -> str:
        config = config or GenerationConfig()
        responses: list[LLMResponse] = []

        critique_text, initial_response = self._generate_initial_critique(explanation, config)
        responses.append(initial_response)

        iterations_used = 0
        for iteration in range(self.max_iterations):
            iterations_used = iteration + 1
            sufficient, review_notes, review_response = self._self_review(
                explanation, critique_text, config
            )
            responses.append(review_response)
            if sufficient:
                break
            critique_text, revise_response = self._revise(
                explanation, critique_text, review_notes, config
            )
            responses.append(revise_response)

        self.last_response = LLMResponse(
            text=critique_text,
            prompt_tokens=sum(r.prompt_tokens for r in responses),
            completion_tokens=sum(r.completion_tokens for r in responses),
            latency_ms=sum(r.latency_ms for r in responses),
        )
        self.last_llm_calls = len(responses)
        self.last_iterations_used = iterations_used
        return critique_text

    def _generate_initial_critique(
        self, explanation: str, config: GenerationConfig
    ) -> tuple[str, LLMResponse]:
        messages = [
            Message(
                role="system",
                content=(
                    "You are a content fidelity critic for a computer science course. "
                    "Evaluate the student's explanation below for factual, conceptual, and procedural "
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
        return response.text, response

    def _self_review(
        self, explanation: str, critique_text: str, config: GenerationConfig
    ) -> tuple[bool, str, LLMResponse]:
        messages = [
            Message(
                role="system",
                content=(
                    "You are reviewing your own critique of a student's technical explanation. "
                    "Check whether your critique was strict enough, whether you missed any factual, "
                    "conceptual, or procedural issue, and whether any claim needs correction. "
                    "Be explicit about weaknesses in the critique.\n\n"
                    "Respond with ONLY a JSON object in this exact format, no other text:\n"
                    '{"sufficient": true or false, "notes": "..."}\n'
                    "Set sufficient=true only if the critique already captures every significant "
                    "issue accurately and needs no changes. Otherwise set sufficient=false and use "
                    "notes to explain exactly what is missing, wrong, or judged too leniently."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"Student explanation:\n\n{explanation}\n\n"
                    f"Your current critique:\n\n{critique_text}"
                ),
            ),
        ]
        response = self.client.generate(messages, config=config)
        sufficient, notes = self._parse_review(response.text)
        return sufficient, notes, response

    def _parse_review(self, text: str) -> tuple[bool, str]:
        data = None
        try:
            data = json.loads(text.strip())
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                try:
                    data = json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    data = None

        if data is None:
            # Couldn't parse a judgment. Fail safe toward "keep improving"
            # rather than silently trusting an unparsed response as done —
            # the outer max_iterations cap still bounds the total cost.
            return False, text

        return bool(data.get("sufficient", False)), str(data.get("notes", ""))

    def _revise(
        self, explanation: str, critique_text: str, review_notes: str, config: GenerationConfig
    ) -> tuple[str, LLMResponse]:
        messages = [
            Message(
                role="system",
                content=(
                    "You are revising a critique of a student's technical explanation. "
                    "Produce the final critique by incorporating valid self-review findings. "
                    "Keep all legitimate concerns, add any missed issues, and remove unsupported ones. "
                    "Do not mention the self-review process."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"Student explanation:\n\n{explanation}\n\n"
                    f"Current critique:\n\n{critique_text}\n\n"
                    f"Self-review notes:\n\n{review_notes}\n\n"
                    "Write the revised critique."
                ),
            ),
        ]
        response = self.client.generate(messages, config=config)
        return response.text, response
