import json

from fletcher.llm.client import GenerationConfig, LLMClient, LLMResponse
from fletcher.llm.message import Message

MAX_ITERATIONS = 3

INITIAL_CRITIQUE_SYSTEM_PROMPT = (
    "You are a content fidelity critic for a computer science course. "
    "Evaluate the student's explanation below for factual, conceptual, and procedural "
    "accuracy. Point out any misconceptions or missing key ideas. "
    "Do not evaluate English proficiency or writing style."
)


class SelfCritique1Pass:
    """Architecture 1a — the literal README 4.1 baseline: ONE critique pass,
    no self-review or revision. This is the "single-prompt critique" that H1
    compares debate (round >= 1) against, so it must stay exactly one call.
    """

    def __init__(self, client: LLMClient, verbose: bool = False):
        self.client = client
        self.verbose = verbose
        self.last_response: LLMResponse | None = None
        self.last_llm_calls: int = 0

    def critique(self, explanation: str, config: GenerationConfig | None = None) -> str:
        config = config or GenerationConfig()
        messages = [
            Message(role="system", content=INITIAL_CRITIQUE_SYSTEM_PROMPT),
            Message(role="user", content=f"Student explanation:\n\n{explanation}"),
        ]
        self._log("Generating single-pass critique.")
        response = self.client.generate(messages, config=config)
        self.last_response = response
        self.last_llm_calls = 1
        self._log(f"Done. prompt_tokens={response.prompt_tokens} completion_tokens={response.completion_tokens}")
        if self.verbose:
            print(flush=True)
        return response.text

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[self_critique_1pass] {message}", flush=True)


class SelfCritique:
    """Architecture 1b — adaptive self-critique: initial critique, then a
    self-review that judges whether the critique is already sufficient. If
    not, revise and review again, up to `max_iterations`.
    """

    def __init__(self, client: LLMClient, max_iterations: int = MAX_ITERATIONS, verbose: bool = False):
        self.client = client
        self.max_iterations = max_iterations
        self.verbose = verbose
        self._initial_critic = SelfCritique1Pass(client, verbose=False)
        self.last_response: LLMResponse | None = None
        self.last_llm_calls: int = 0
        self.last_iterations_used: int = 0
        # Diagnostics for the last critique() call: how many review steps
        # produced parseable JSON vs. fell back to the fail-safe default.
        self.last_review_parse_failures: int = 0
        self.last_review_parse_successes: int = 0

    def critique(self, explanation: str, config: GenerationConfig | None = None) -> str:
        config = config or GenerationConfig()
        responses: list[LLMResponse] = []
        self.last_review_parse_failures = 0
        self.last_review_parse_successes = 0

        critique_text = self._initial_critic.critique(explanation, config=config)
        responses.append(self._initial_critic.last_response)
        self._log("[iteration 0] Initial critique generated.")

        iterations_used = 0
        for iteration in range(self.max_iterations):
            iterations_used = iteration + 1
            sufficient, review_notes, review_response, parsed_ok = self._self_review(
                explanation, critique_text, config
            )
            responses.append(review_response)
            if parsed_ok:
                self.last_review_parse_successes += 1
            else:
                self.last_review_parse_failures += 1

            self._log(
                f"[iteration {iterations_used}] review parsed={parsed_ok} "
                f"sufficient={sufficient} notes={review_notes[:120]!r}"
            )

            if sufficient:
                self._log(f"[iteration {iterations_used}] Sufficient — stopping early.")
                break

            critique_text, revise_response = self._revise(
                explanation, critique_text, review_notes, config
            )
            responses.append(revise_response)
            self._log(f"[iteration {iterations_used}] Revised critique.")
        else:
            self._log(f"Reached max_iterations={self.max_iterations} without a sufficient verdict.")

        if self.verbose:
            print(flush=True)

        self.last_response = LLMResponse(
            text=critique_text,
            prompt_tokens=sum(r.prompt_tokens for r in responses),
            completion_tokens=sum(r.completion_tokens for r in responses),
            latency_ms=sum(r.latency_ms for r in responses),
        )
        self.last_llm_calls = len(responses)
        self.last_iterations_used = iterations_used
        return critique_text

    def _self_review(
        self, explanation: str, critique_text: str, config: GenerationConfig
    ) -> tuple[bool, str, LLMResponse, bool]:
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
        sufficient, notes, parsed_ok = self._parse_review(response.text)
        return sufficient, notes, response, parsed_ok

    def _parse_review(self, text: str) -> tuple[bool, str, bool]:
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
            return False, text, False

        return bool(data.get("sufficient", False)), str(data.get("notes", "")), True

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

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[self_critique] {message}", flush=True)
