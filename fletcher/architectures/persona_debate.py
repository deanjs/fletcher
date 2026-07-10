from typing import TYPE_CHECKING

from fletcher.agents.content_critic.conceptual import ConceptualCritic
from fletcher.agents.content_critic.completeness import CompletenessCritic
from fletcher.agents.content_critic.procedural import ProceduralCritic
from fletcher.agents.orchestrator import DebateCriticSpec, DebateOrchestrator
from fletcher.llm.client import GenerationConfig, LLMClient

if TYPE_CHECKING:
    from fletcher.rag.lecture_notes.retriever import LectureNoteRetriever

MAX_PERSONA_ROUNDS = 2

PERSONA_CRITIC_CLASSES = {
    "conceptual": ConceptualCritic,
    "procedural": ProceduralCritic,
    "completeness": CompletenessCritic,
}


def _make_critic(critic_cls, client: LLMClient, persona: str, retriever):
    if critic_cls is CompletenessCritic:
        return critic_cls(client, retriever=retriever)
    return critic_cls(client, persona=persona, retriever=retriever)


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
    """Backward-compatible wrapper around DebateOrchestrator."""
    orchestrator = DebateOrchestrator.from_critic_map(
        critics,
        max_rounds=max_rounds,
        config=config,
        verbose=verbose,
        log_prefix=log_prefix,
    )
    return orchestrator.run(explanation)


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
    specs: list[DebateCriticSpec] = []
    for role, persona in persona_list:
        critic_cls = PERSONA_CRITIC_CLASSES.get(role)
        if critic_cls is None:
            raise ValueError(f"Unsupported role for persona debate: {role}")
        key = _debate_key(role, persona)
        persona_client = client_per_key.get(key, client)
        specs.append(
            DebateCriticSpec(
                key=key,
                role=role,
                persona=persona,
                critic=_make_critic(critic_cls, persona_client, persona, retriever),
            )
        )

    orchestrator = DebateOrchestrator(
        specs,
        max_rounds=max_rounds,
        config=config,
        verbose=verbose,
        log_prefix="persona_debate",
        axis_config={"N": [{"role": role, "persona": persona} for role, persona in persona_list]},
    )
    return orchestrator.run(explanation)


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

    specs: list[DebateCriticSpec] = []
    for label, model_client in client_list:
        key = _model_debate_key(role, persona, label)
        specs.append(
            DebateCriticSpec(
                key=key,
                role=role,
                persona=persona,
                model=label,
                critic=_make_critic(critic_cls, model_client, persona, retriever),
            )
        )

    orchestrator = DebateOrchestrator(
        specs,
        max_rounds=max_rounds,
        config=config,
        verbose=verbose,
        log_prefix="model_debate",
        axis_config={"M": [label for label, _ in client_list], "role": role, "persona": persona},
    )
    return orchestrator.run(explanation)


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
    debate_specs: list[DebateCriticSpec] = []
    for role, persona, label, model_client in specs:
        critic_cls = PERSONA_CRITIC_CLASSES.get(role)
        if critic_cls is None:
            raise ValueError(f"Unsupported role for combined debate: {role}")
        key = _model_debate_key(role, persona, label)
        debate_specs.append(
            DebateCriticSpec(
                key=key,
                role=role,
                persona=persona,
                model=label,
                critic=_make_critic(critic_cls, model_client, persona, retriever),
            )
        )

    orchestrator = DebateOrchestrator(
        debate_specs,
        max_rounds=max_rounds,
        config=config,
        verbose=verbose,
        log_prefix="combined_debate",
        axis_config={
            "NM": [
                {"role": role, "persona": persona, "model": label}
                for role, persona, label, _ in specs
            ]
        },
    )
    return orchestrator.run(explanation)
