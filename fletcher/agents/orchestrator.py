from dataclasses import dataclass
from typing import Any

from fletcher.llm.client import GenerationConfig


@dataclass(frozen=True)
class DebateCriticSpec:
    key: str
    role: str
    critic: Any
    persona: str | None = None
    model: str | None = None


class DebateOrchestrator:
    """Runs same-target critic debate and emits a distillation-friendly log."""

    def __init__(
        self,
        critics: list[DebateCriticSpec],
        max_rounds: int,
        config: GenerationConfig | None = None,
        verbose: bool = False,
        log_prefix: str = "debate",
        axis_config: dict | None = None,
    ):
        if not critics:
            raise ValueError("DebateOrchestrator requires at least one critic.")
        self.critics = critics
        self.max_rounds = max_rounds
        self.config = config
        self.verbose = verbose
        self.log_prefix = log_prefix
        self.axis_config = axis_config or {}

    @classmethod
    def from_critic_map(
        cls,
        critics: dict[str, tuple[str, object]],
        max_rounds: int,
        config: GenerationConfig | None = None,
        verbose: bool = False,
        log_prefix: str = "debate",
        axis_config: dict | None = None,
    ) -> "DebateOrchestrator":
        specs = [
            DebateCriticSpec(key=key, role=role, critic=critic)
            for key, (role, critic) in critics.items()
        ]
        return cls(
            specs,
            max_rounds=max_rounds,
            config=config,
            verbose=verbose,
            log_prefix=log_prefix,
            axis_config=axis_config,
        )

    def run(self, explanation: str, metadata: dict | None = None) -> dict:
        debate_history: list[dict] = []
        latest_verdicts: dict[str, Any] = {}
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_llm_calls = 0
        total_retrieval_ms = 0.0
        total_retrieved_skill_count = 0
        total_retrieved_skill_tokens = 0
        rounds_run = 0

        self._log_topic(explanation)

        for round_index in range(self.max_rounds):
            rounds_run = round_index + 1
            round_verdicts: dict[str, dict] = {}

            for spec in self.critics:
                verdict = spec.critic.evaluate(
                    explanation,
                    config=self.config,
                    debate_history=debate_history,
                    debate_key=spec.key,
                    request_message=True,
                )

                self._log_verdict(spec.key, rounds_run, verdict)
                round_verdicts[spec.key] = self._verdict_record(spec, verdict)
                latest_verdicts[spec.key] = verdict

                response = spec.critic.last_response
                if response:
                    total_prompt_tokens += response.prompt_tokens
                    total_completion_tokens += response.completion_tokens
                    total_llm_calls += 1
                skill_metrics = self._skill_metrics(spec.critic)
                total_retrieval_ms += skill_metrics["retrieval_ms"]
                total_retrieved_skill_count += skill_metrics["retrieved_skill_count"]
                total_retrieved_skill_tokens += skill_metrics["retrieved_skill_tokens"]

            disagreement = self._has_disagreement(latest_verdicts)
            debate_history.append({
                "round": rounds_run,
                "verdicts": round_verdicts,
                "disagreement": disagreement,
            })

            if not disagreement:
                if self.verbose:
                    print(f"[{self.log_prefix}] Unanimous after round {rounds_run}, stopping early.", flush=True)
                break

            if self.verbose:
                print(flush=True)

        if self.verbose:
            print(flush=True)

        flagged_count = sum(1 for verdict in latest_verdicts.values() if verdict.flagged)
        not_flagged_count = len(latest_verdicts) - flagged_count
        final_flagged = flagged_count > not_flagged_count
        consensus = not self._has_disagreement(latest_verdicts)

        metrics = {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "llm_calls": total_llm_calls,
            "retrieval_ms": total_retrieval_ms,
            "retrieved_skill_count": total_retrieved_skill_count,
            "retrieved_skill_tokens": total_retrieved_skill_tokens,
        }
        final = {
            "flagged": final_flagged,
            "decision_rule": "majority_vote",
            "consensus": consensus,
            "rounds_run": rounds_run,
            "flagged_count": flagged_count,
            "not_flagged_count": not_flagged_count,
        }
        debate_log = {
            "architecture": "arch2",
            "axis_config": self.axis_config,
            "input": {"explanation": explanation},
            "metadata": metadata or {},
            "rounds": debate_history,
            "final": final,
            "metrics": metrics,
        }

        return {
            "flagged": final_flagged,
            "rounds_run": rounds_run,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "llm_calls": total_llm_calls,
            "debate_history": debate_history,
            "debate_log": debate_log,
            "consensus": consensus,
            "decision_rule": "majority_vote",
        }

    def _log_topic(self, explanation: str) -> None:
        if not self.verbose:
            return
        topic = explanation if len(explanation) <= 160 else explanation[:157] + "..."
        print(f"[{self.log_prefix}] Debate topic: \"{topic}\"", flush=True)
        print(flush=True)

    def _log_verdict(self, key: str, round_number: int, verdict: Any) -> None:
        if not self.verbose:
            return
        recipients = [spec.key for spec in self.critics if spec.key != key]
        print(
            f"[{self.log_prefix}][{key}][round {round_number}] "
            f"flagged={verdict.flagged} confidence={verdict.confidence:.2f}",
            flush=True,
        )
        print(f"    reasoning: {verdict.reasoning}", flush=True)
        print(f"    [{key} -> {', '.join(recipients)}]: {verdict.message_to_others}", flush=True)

    def _verdict_record(self, spec: DebateCriticSpec, verdict: Any) -> dict:
        record = verdict.model_dump()
        record.setdefault("role", spec.role)
        if spec.persona is not None:
            record["persona"] = spec.persona
        if spec.model is not None:
            record["model"] = spec.model
        skill_metrics = self._skill_metrics(spec.critic)
        if skill_metrics["retrieved_skill_count"] > 0:
            record["skill_metrics"] = skill_metrics
        return record

    def _has_disagreement(self, verdicts: dict[str, Any]) -> bool:
        return len({verdict.flagged for verdict in verdicts.values()}) > 1

    def _skill_metrics(self, critic: Any) -> dict:
        return {
            "retrieval_ms": float(getattr(critic, "last_skill_metrics", {}).get("retrieval_ms", 0.0)),
            "retrieved_skill_count": int(
                getattr(critic, "last_skill_metrics", {}).get("retrieved_skill_count", 0)
            ),
            "retrieved_skill_tokens": int(
                getattr(critic, "last_skill_metrics", {}).get("retrieved_skill_tokens", 0)
            ),
            "retrieved_skill_ids": list(
                getattr(critic, "last_skill_metrics", {}).get("retrieved_skill_ids", [])
            ),
        }
