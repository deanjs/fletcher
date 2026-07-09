from typing import TYPE_CHECKING

from fletcher.agents.content_critic.conceptual import ConceptualCritic
from fletcher.agents.content_critic.procedural import ProceduralCritic
from fletcher.llm.client import GenerationConfig, LLMClient

if TYPE_CHECKING:
    from fletcher.rag.lecture_notes.retriever import LectureNoteRetriever

MAX_PERSONA_ROUNDS = 2

PERSONA_CRITIC_CLASSES = {
    "conceptual": ConceptualCritic,
    "procedural": ProceduralCritic,
}


def _debate_key(role: str, persona: str) -> str:
    return f"{role}_{persona}"


def run_persona_debate(
    client: LLMClient,
    persona_list: list[tuple[str, str]],
    explanation: str,
    config: GenerationConfig | None = None,
    retriever: "LectureNoteRetriever | None" = None,
    max_rounds: int = MAX_PERSONA_ROUNDS,
    verbose: bool = False,
    client_per_key: dict[str, LLMClient] | None = None,
) -> dict:
    """Run an N-axis debate: several personas of the SAME role evaluate the
    same explanation and, if they disagree, see each other's verdicts and
    re-evaluate for up to `max_rounds`. This mirrors the R-axis debate in
    `fletcher.architectures.debate`, but the axis of disagreement is persona
    (strict/neutral/merciful) rather than role (conceptual/procedural/completeness).

    `client_per_key` optionally maps a debate key (e.g. "conceptual_merciful",
    see `_debate_key`) to a different LLMClient, so individual personas can be
    backed by a different model instead of all personas sharing one model.
    """
    client_per_key = client_per_key or {}
    critics: dict[str, tuple[str, object]] = {}
    for role, persona in persona_list:
        critic_cls = PERSONA_CRITIC_CLASSES.get(role)
        if critic_cls is None:
            raise ValueError(f"Unsupported role for persona debate: {role}")
        key = _debate_key(role, persona)
        persona_client = client_per_key.get(key, client)
        critics[key] = (role, critic_cls(persona_client, persona=persona, retriever=retriever))

    debate_history: list[dict] = []
    latest_verdicts: dict[str, object] = {}
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_llm_calls = 0
    rounds_run = 0

    for round_index in range(max_rounds):
        rounds_run = round_index + 1
        round_verdicts: dict[str, dict] = {}
        round_debate_turns: dict[str, str] = {}

        for key, (role, critic) in critics.items():
            verdict = critic.evaluate(
                explanation,
                config=config,
                debate_history=debate_history,
                debate_key=key,
            )
            debate_text = critic.compose_debate_turn(
                explanation,
                verdict,
                debate_history=debate_history,
                config=config,
                debate_key=key,
            )

            if verbose:
                print(
                    f"[persona_debate][{key}][round {rounds_run}] "
                    f"flagged={verdict.flagged} confidence={verdict.confidence:.2f}",
                    flush=True,
                )

            round_verdicts[key] = verdict.model_dump()
            round_debate_turns[key] = debate_text
            latest_verdicts[key] = verdict

            response = critic.last_response
            debate_response = critic.last_debate_response
            if response:
                total_prompt_tokens += response.prompt_tokens
                total_completion_tokens += response.completion_tokens
                total_llm_calls += 1
            if debate_response:
                total_prompt_tokens += debate_response.prompt_tokens
                total_completion_tokens += debate_response.completion_tokens
                total_llm_calls += 1

        debate_history.append(
            {"round": rounds_run, "verdicts": round_verdicts, "debate_turns": round_debate_turns}
        )

        flags = {v.flagged for v in latest_verdicts.values()}
        if len(flags) == 1:
            if verbose:
                print(f"[persona_debate] Unanimous after round {rounds_run}, stopping early.", flush=True)
            break

    flagged_count = sum(1 for v in latest_verdicts.values() if v.flagged)
    not_flagged_count = len(latest_verdicts) - flagged_count
    final_flagged = flagged_count > not_flagged_count

    return {
        "flagged": final_flagged,
        "rounds_run": rounds_run,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "llm_calls": total_llm_calls,
        "debate_history": debate_history,
    }
