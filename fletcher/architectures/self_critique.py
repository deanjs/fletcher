import json
import re
from dataclasses import replace

from fletcher.llm.client import GenerationConfig, LLMClient, LLMResponse
from fletcher.llm.message import Message

MAX_ITERATIONS = 6

# The model reports a raw sufficiency CONFIDENCE (0.0-1.0); whether that
# counts as "done" is a separate decision rule applied in code, not baked
# into the model's own true/false call. Phase 1 (now): a fixed threshold you
# tune by hand. Phase 2 (later): replace this threshold with a learned
# policy (same "manual first, RL later" progression already used for K).
SUFFICIENCY_THRESHOLD = 0.8

# Review responses are JSON with a free-text "notes" field; at max_new_tokens
# this occasionally gets cut off mid-string before the closing brace, which
# silently fails to parse and forces the fail-safe "insufficient" path every
# time regardless of what the model actually judged. Give review calls a
# larger budget than the default critique length.
REVIEW_MIN_MAX_NEW_TOKENS = 400

VERDICT_LINE_INSTRUCTION = (
    "After the critique, end your response with exactly one line, on its own, "
    "in exactly this format (no other text on that line): "
    "\"VERDICT: FLAGGED\" if you identified any factual, conceptual, or "
    "procedural issue, or \"VERDICT: OK\" if the explanation is accurate and "
    "complete."
)

INITIAL_CRITIQUE_SYSTEM_PROMPT = (
    "You are a content fidelity critic for a computer science course. "
    "Evaluate the student's explanation below for factual, conceptual, and procedural "
    "accuracy. Point out any misconceptions or missing key ideas. "
    "Do not evaluate English proficiency or writing style.\n\n"
    f"{VERDICT_LINE_INSTRUCTION}"
)

_VERDICT_LINE_RE = re.compile(r"^\s*VERDICT:\s*(FLAGGED|OK)\s*$", re.IGNORECASE | re.MULTILINE)

# Fallback only: used when the model doesn't emit a parseable VERDICT line.
_FALLBACK_KEYWORDS = ["incorrect", "wrong", "missing", "error", "inaccurate", "issue"]


def _parse_verdict(text: str) -> tuple[str, bool]:
    """Strip the trailing VERDICT line (if present) and return
    (critique_text_without_verdict_line, flagged). Falls back to a crude
    keyword scan when the model didn't follow the VERDICT line format —
    keeps the caller from crashing, but that fallback is a last resort, not
    the primary signal.
    """
    match = _VERDICT_LINE_RE.search(text)
    if match:
        flagged = match.group(1).upper() == "FLAGGED"
        clean_text = (text[: match.start()] + text[match.end():]).strip()
        return clean_text, flagged

    flagged = any(keyword in text.lower() for keyword in _FALLBACK_KEYWORDS)
    return text, flagged


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
        self.last_flagged: bool = False

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
        critique_text, self.last_flagged = _parse_verdict(response.text)
        self._log(
            f"Done. flagged={self.last_flagged} "
            f"prompt_tokens={response.prompt_tokens} completion_tokens={response.completion_tokens}"
        )
        if self.verbose:
            print(flush=True)
        return critique_text

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[self_critique_1pass] {message}", flush=True)


class SelfCritique:
    """Architecture 1b — adaptive self-critique: initial critique, then a
    self-review that reports a sufficiency CONFIDENCE (0.0-1.0). Whether that
    counts as "done" is decided in code via `sufficiency_threshold`, not by
    the model directly — see SUFFICIENCY_THRESHOLD above. Below threshold:
    revise and review again, up to `max_iterations`.
    """

    def __init__(
        self,
        client: LLMClient,
        max_iterations: int = MAX_ITERATIONS,
        sufficiency_threshold: float = SUFFICIENCY_THRESHOLD,
        verbose: bool = False,
    ):
        self.client = client
        self.max_iterations = max_iterations
        self.sufficiency_threshold = sufficiency_threshold
        self.verbose = verbose
        self._initial_critic = SelfCritique1Pass(client, verbose=False)
        self.last_response: LLMResponse | None = None
        self.last_llm_calls: int = 0
        self.last_iterations_used: int = 0
        self.last_flagged: bool = False
        # Diagnostics for the last critique() call: how many review steps
        # produced parseable JSON vs. fell back to the fail-safe default,
        # plus the confidence value from the final review.
        self.last_review_parse_failures: int = 0
        self.last_review_parse_successes: int = 0
        self.last_confidence: float = 0.0

    def critique(self, explanation: str, config: GenerationConfig | None = None) -> str:
        config = config or GenerationConfig()
        responses: list[LLMResponse] = []
        self.last_review_parse_failures = 0
        self.last_review_parse_successes = 0

        critique_text = self._initial_critic.critique(explanation, config=config)
        flagged = self._initial_critic.last_flagged
        responses.append(self._initial_critic.last_response)
        self._log("[iteration 0] Initial critique generated.")

        iterations_used = 0
        for iteration in range(self.max_iterations):
            iterations_used = iteration + 1
            confidence, review_notes, review_response, parsed_ok = self._self_review(
                explanation, critique_text, config
            )
            responses.append(review_response)
            self.last_confidence = confidence
            if parsed_ok:
                self.last_review_parse_successes += 1
            else:
                self.last_review_parse_failures += 1

            sufficient = confidence >= self.sufficiency_threshold
            self._log(
                f"[iteration {iterations_used}] review parsed={parsed_ok} "
                f"confidence={confidence:.2f} threshold={self.sufficiency_threshold:.2f} "
                f"sufficient={sufficient} notes={review_notes[:120]!r}"
            )

            if sufficient:
                self._log(f"[iteration {iterations_used}] Sufficient — stopping early.")
                break

            critique_text, flagged, revise_response = self._revise(
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
        self.last_flagged = flagged
        return critique_text

    def _self_review(
        self, explanation: str, critique_text: str, config: GenerationConfig
    ) -> tuple[float, str, LLMResponse, bool]:
        review_config = replace(
            config, max_new_tokens=max(config.max_new_tokens, REVIEW_MIN_MAX_NEW_TOKENS)
        )
        messages = [
            Message(
                role="system",
                content=(
                    "You are reviewing your own critique of a student's technical explanation. "
                    "Check whether your critique was strict enough, whether you missed any factual, "
                    "conceptual, or procedural issue, and whether any claim needs correction. "
                    "Be explicit about weaknesses in the critique.\n\n"
                    "Respond with ONLY a JSON object in this exact format, no other text:\n"
                    '{"confidence": 0.0 to 1.0, "notes": "..."}\n'
                    "confidence is how sure you are that the critique ALREADY captures every "
                    "significant issue accurately and needs no changes — 1.0 means completely "
                    "sure it's done, 0.0 means it clearly still needs work. Use notes to explain "
                    "exactly what is missing, wrong, or judged too leniently (leave brief if "
                    "confidence is high). Keep notes to at most two short sentences so the JSON "
                    "stays short and complete."
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
        response = self.client.generate(messages, config=review_config)
        confidence, notes, parsed_ok = self._parse_review(response.text)
        return confidence, notes, response, parsed_ok

    def _parse_review(self, text: str) -> tuple[float, str, bool]:
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

        if data is None or "confidence" not in data:
            # Couldn't parse a judgment. Fail safe toward "keep improving"
            # (confidence=0.0, always below the threshold) rather than
            # silently trusting an unparsed response as done — the outer
            # max_iterations cap still bounds the total cost.
            return 0.0, text, False

        try:
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
        except (TypeError, ValueError):
            return 0.0, text, False

        return confidence, str(data.get("notes", "")), True

    def _revise(
        self, explanation: str, critique_text: str, review_notes: str, config: GenerationConfig
    ) -> tuple[str, bool, LLMResponse]:
        messages = [
            Message(
                role="system",
                content=(
                    "You are revising a critique of a student's technical explanation. "
                    "Produce the final critique by incorporating valid self-review findings. "
                    "Keep all legitimate concerns, add any missed issues, and remove unsupported ones. "
                    "Do not mention the self-review process.\n\n"
                    f"{VERDICT_LINE_INSTRUCTION}"
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
        revised_text, flagged = _parse_verdict(response.text)
        return revised_text, flagged, response

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[self_critique] {message}", flush=True)
