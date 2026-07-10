from typing import TYPE_CHECKING

from fletcher.agents.schemas import CriticVerdict
from fletcher.agents.synthesizer import Synthesizer
from fletcher.architectures.persona_debate import run_persona_debate
from fletcher.llm.client import GenerationConfig, LLMClient

if TYPE_CHECKING:
    from fletcher.rag.lecture_notes.retriever import LectureNoteRetriever


DEFAULT_PERSONAS = ["strict", "merciful"]


def run_full_debate(
    client: LLMClient,
    roles: list[str],
    explanation: str,
    personas_by_role: dict[str, list[str]] | None = None,
    config: GenerationConfig | None = None,
    retriever: "LectureNoteRetriever | None" = None,
    retriever_per_role: dict[str, "LectureNoteRetriever"] | None = None,
    max_rounds: int = 2,
    verbose: bool = False,
) -> dict:
    """Run Architecture 2 as a complete two-stage pipeline.

    Stage 1 runs same-target persona debate inside each role. Stage 2 combines
    the final per-role verdicts without cross-role rebuttal.
    """

    personas_by_role = personas_by_role or {}
    retriever_per_role = retriever_per_role or {}
    role_results: dict[str, dict] = {}
    role_verdicts: dict[str, CriticVerdict] = {}
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_llm_calls = 0

    for role in roles:
        personas = personas_by_role.get(role, DEFAULT_PERSONAS)
        persona_list = [(role, persona) for persona in personas]
        role_retriever = retriever_per_role.get(role, retriever)
        result = run_persona_debate(
            client,
            persona_list,
            explanation,
            config=config,
            retriever=role_retriever,
            max_rounds=max_rounds,
            verbose=verbose,
        )
        role_results[role] = result
        total_prompt_tokens += result["prompt_tokens"]
        total_completion_tokens += result["completion_tokens"]
        total_llm_calls += result["llm_calls"]
        role_verdicts[role] = _majority_verdict(role, result)

    synthesized = Synthesizer().synthesize(role_verdicts)
    debate_log = {
        "architecture": "arch2_full",
        "axis_config": {
            "R": roles,
            "N": personas_by_role or {role: DEFAULT_PERSONAS for role in roles},
            "K": max(0, max_rounds - 1),
        },
        "input": {"explanation": explanation},
        "roles": {
            role: result["debate_log"]
            for role, result in role_results.items()
        },
        "final": {
            "flagged": synthesized["issue_found"],
            "decision_rule": "role_synthesis_after_per_role_majority_vote",
            "flagged_roles": synthesized["flagged_roles"],
            "final_result": synthesized["final_result"],
        },
        "metrics": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "llm_calls": total_llm_calls,
        },
    }

    return {
        "flagged": synthesized["issue_found"],
        "issue_found": synthesized["issue_found"],
        "final_result": synthesized["final_result"],
        "role_results": role_results,
        "role_verdicts": {role: verdict.model_dump() for role, verdict in role_verdicts.items()},
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "llm_calls": total_llm_calls,
        "debate_log": debate_log,
    }


def _majority_verdict(role: str, debate_result: dict) -> CriticVerdict:
    latest_round = debate_result["debate_history"][-1]
    verdicts = latest_round["verdicts"]
    flagged = debate_result["flagged"]
    matching = [
        verdict
        for verdict in verdicts.values()
        if verdict["flagged"] == flagged
    ]
    source = matching[0] if matching else next(iter(verdicts.values()))
    return CriticVerdict(
        role=role,
        flagged=flagged,
        confidence=float(source.get("confidence", 0.0)),
        reasoning=str(source.get("reasoning", "")),
        message_to_others=str(source.get("message_to_others", "")),
    )
