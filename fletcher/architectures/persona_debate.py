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


def _model_debate_key(role: str, persona: str, label: str) -> str:
    return f"{role}_{persona}_{label}"


def _run_ensemble_debate(
    critics: dict[str, tuple[str, object]],
    explanation: str,
    config: GenerationConfig | None,
    max_rounds: int,
    verbose: bool,
    log_prefix: str,
) -> dict:
    """Shared Stage-1 engine: N critics sharing the same role evaluate the
    same explanation and, if they disagree, see each other's verdicts and
    re-evaluate for up to `max_rounds`, stopping early once unanimous.
    Used both for persona diversity (N-axis) and model diversity (M-axis) —
    the only difference between the two is how `critics` keys are built.
    """
    debate_history: list[dict] = []
    latest_verdicts: dict[str, object] = {}
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_llm_calls = 0
    rounds_run = 0

    if verbose:
        topic = explanation if len(explanation) <= 160 else explanation[:157] + "..."
        print(f"[{log_prefix}] Debate topic: \"{topic}\"", flush=True)
        print(flush=True)

    for round_index in range(max_rounds):
        rounds_run = round_index + 1
        round_verdicts: dict[str, dict] = {}

        for key, (role, critic) in critics.items():
            verdict = critic.evaluate(
                explanation,
                config=config,
                debate_history=debate_history,
                debate_key=key,
                request_message=True,
            )

            if verbose:
                recipients = [other_key for other_key in critics if other_key != key]
                print(
                    f"[{log_prefix}][{key}][round {rounds_run}] "
                    f"flagged={verdict.flagged} confidence={verdict.confidence:.2f}",
                    flush=True,
                )
                print(f"    reasoning: {verdict.reasoning}", flush=True)
                print(f"    [{key} -> {', '.join(recipients)}]: {verdict.message_to_others}", flush=True)

            round_verdicts[key] = verdict.model_dump()
            latest_verdicts[key] = verdict

            response = critic.last_response
            if response:
                total_prompt_tokens += response.prompt_tokens
                total_completion_tokens += response.completion_tokens
                total_llm_calls += 1

        debate_history.append({"round": rounds_run, "verdicts": round_verdicts})

        flags = {v.flagged for v in latest_verdicts.values()}
        if len(flags) == 1:
            if verbose:
                print(f"[{log_prefix}] Unanimous after round {rounds_run}, stopping early.", flush=True)
            break

        if verbose:
            print(flush=True)

    if verbose:
        print(flush=True)

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
    """N-axis Stage-1 debate: several PERSONAS of the SAME role, SAME model
    by default, evaluate the same explanation and debate for up to
    `max_rounds` if they disagree.

    `client_per_key` optionally overrides the model for one debate key (e.g.
    "conceptual_merciful") — kept only for backward compatibility / ad-hoc
    experiments. For a controlled model-diversity study use
    `run_model_debate` instead, which holds persona fixed so the model is the
    only thing that varies.
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

    return _run_ensemble_debate(
        critics, explanation, config, max_rounds, verbose, log_prefix="persona_debate"
    )


def run_model_debate(
    role: str,
    persona: str,
    client_list: list[tuple[str, LLMClient]],
    explanation: str,
    config: GenerationConfig | None = None,
    retriever: "LectureNoteRetriever | None" = None,
    max_rounds: int = MAX_PERSONA_ROUNDS,
    verbose: bool = False,
) -> dict:
    """M-axis Stage-1 debate: several MODELS evaluate the SAME role with the
    SAME (fixed) persona, and debate for up to `max_rounds` if they disagree.
    This isolates model diversity from persona diversity — the only thing
    that varies between critics here is which weights are answering.

    `client_list` is a list of (label, LLMClient) pairs, e.g.
    [("qwen", qwen_client), ("llama", llama_client)]. Labels only need to be
    unique within the call; they show up in verbose logs and debate_history.
    """
    critic_cls = PERSONA_CRITIC_CLASSES.get(role)
    if critic_cls is None:
        raise ValueError(f"Unsupported role for model debate: {role}")

    critics: dict[str, tuple[str, object]] = {}
    for label, model_client in client_list:
        key = _model_debate_key(role, persona, label)
        critics[key] = (role, critic_cls(model_client, persona=persona, retriever=retriever))

    return _run_ensemble_debate(
        critics, explanation, config, max_rounds, verbose, log_prefix="model_debate"
    )


def run_combined_debate(
    specs: list[tuple[str, str, str, LLMClient]],
    explanation: str,
    config: GenerationConfig | None = None,
    retriever: "LectureNoteRetriever | None" = None,
    max_rounds: int = MAX_PERSONA_ROUNDS,
    verbose: bool = False,
) -> dict:
    """NM-axis Stage-1 debate: persona AND model both vary at once.

    Only use this AFTER N (persona alone) and M (model alone) have each been
    validated independently — with both axes varying together, any observed
    disagreement or accuracy change can no longer be attributed to persona
    vs. model specifically. This is the deliberate follow-up combination
    experiment, not a substitute for the isolated N/M sweeps.

    `specs` is a list of (role, persona, label, client) tuples, e.g.
    [("conceptual", "strict", "qwen", qwen_client),
     ("conceptual", "merciful", "llama", llama_client)].
    `label` only needs to be unique per role+persona pair (matters when the
    same persona is reused with a different model).
    """
    critics: dict[str, tuple[str, object]] = {}
    for role, persona, label, model_client in specs:
        critic_cls = PERSONA_CRITIC_CLASSES.get(role)
        if critic_cls is None:
            raise ValueError(f"Unsupported role for combined debate: {role}")
        key = _model_debate_key(role, persona, label)
        critics[key] = (role, critic_cls(model_client, persona=persona, retriever=retriever))

    return _run_ensemble_debate(
        critics, explanation, config, max_rounds, verbose, log_prefix="combined_debate"
    )
