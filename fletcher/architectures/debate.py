from typing import TYPE_CHECKING, TypedDict

from langgraph.graph import END, StateGraph

from fletcher.agents.content_critic.conceptual import ConceptualCritic
from fletcher.agents.content_critic.procedural import ProceduralCritic
from fletcher.agents.synthesizer import Synthesizer
from fletcher.llm.client import GenerationConfig, LLMClient

if TYPE_CHECKING:
    from fletcher.rag.lecture_notes.retriever import LectureNoteRetriever


ALL_ROLES = ["conceptual", "procedural", "completeness"]

# R-axis (this module) is Stage 2 in README 4.2: different roles evaluate
# different targets (concepts vs. procedure vs. completeness), so there is
# nothing for them to "disagree" about — a procedural error and a conceptual
# error can both be true at once. Stage 2 is therefore a single synthesis
# pass with NO rebuttal rounds. K (debate depth) only applies to Stage 1 —
# critics sharing the same role and the same evaluation target (see
# fletcher.architectures.persona_debate for the N-axis / persona version of
# Stage 1, and the model-diversity version to follow).


class DebateState(TypedDict):
    explanation: str
    conceptual_verdict: dict
    procedural_verdict: dict
    completeness_verdict: dict
    issue_found: bool
    final_result: str
    active_roles: list[str]
    total_prompt_tokens: int
    total_completion_tokens: int
    total_llm_calls: int
    verbose: bool


def make_conceptual_critic_node(
    client: LLMClient,
    persona: str = "neutral",
    retriever: "LectureNoteRetriever | None" = None,
    config: GenerationConfig | None = None,
):
    critic = ConceptualCritic(client, persona=persona, retriever=retriever)

    def node(state: DebateState) -> dict:
        _log_state(state, "conceptual", "Evaluating the explanation.")
        verdict = critic.evaluate(state["explanation"], config=config)
        _log_state(
            state,
            "conceptual",
            f"Verdict flagged={verdict.flagged} confidence={verdict.confidence:.2f}.",
        )
        response = critic.last_response
        prompt_tokens = response.prompt_tokens if response else 0
        completion_tokens = response.completion_tokens if response else 0
        llm_calls = 1 if response else 0
        return {
            "conceptual_verdict": verdict.model_dump(),
            "total_prompt_tokens": state["total_prompt_tokens"] + prompt_tokens,
            "total_completion_tokens": state["total_completion_tokens"] + completion_tokens,
            "total_llm_calls": state["total_llm_calls"] + llm_calls,
        }

    return node


def make_procedural_critic_node(
    client: LLMClient,
    persona: str = "neutral",
    retriever: "LectureNoteRetriever | None" = None,
    config: GenerationConfig | None = None,
):
    critic = ProceduralCritic(client, persona=persona, retriever=retriever)

    def node(state: DebateState) -> dict:
        _log_state(state, "procedural", "Evaluating the explanation.")
        verdict = critic.evaluate(state["explanation"], config=config)
        _log_state(
            state,
            "procedural",
            f"Verdict flagged={verdict.flagged} confidence={verdict.confidence:.2f}.",
        )
        response = critic.last_response
        prompt_tokens = response.prompt_tokens if response else 0
        completion_tokens = response.completion_tokens if response else 0
        llm_calls = 1 if response else 0
        return {
            "procedural_verdict": verdict.model_dump(),
            "total_prompt_tokens": state["total_prompt_tokens"] + prompt_tokens,
            "total_completion_tokens": state["total_completion_tokens"] + completion_tokens,
            "total_llm_calls": state["total_llm_calls"] + llm_calls,
        }

    return node


def make_completeness_critic_node(
    client: LLMClient,
    retriever: "LectureNoteRetriever | None",
    config: GenerationConfig | None = None,
):
    from fletcher.agents.content_critic.completeness import CompletenessCritic

    critic = CompletenessCritic(client, retriever)

    def node(state: DebateState) -> dict:
        grounding_note = "with retrieval grounding" if retriever is not None else "without retrieval grounding"
        _log_state(state, "completeness", f"Evaluating the explanation {grounding_note}.")
        verdict = critic.evaluate(state["explanation"], config=config)
        _log_state(
            state,
            "completeness",
            f"Verdict flagged={verdict.flagged} confidence={verdict.confidence:.2f}.",
        )
        response = critic.last_response
        prompt_tokens = response.prompt_tokens if response else 0
        completion_tokens = response.completion_tokens if response else 0
        llm_calls = 1 if response else 0
        return {
            "completeness_verdict": verdict.model_dump(),
            "total_prompt_tokens": state["total_prompt_tokens"] + prompt_tokens,
            "total_completion_tokens": state["total_completion_tokens"] + completion_tokens,
            "total_llm_calls": state["total_llm_calls"] + llm_calls,
        }

    return node


def make_synthesizer_node(roles: list[str]):
    synthesizer = Synthesizer()

    def node(state: DebateState) -> dict:
        _log_state(state, "synthesizer", "Combining per-role verdicts into the final result (no rebuttal).")

        verdicts = {
            role: state.get(f"{role}_verdict", {})
            for role in roles
        }
        result = synthesizer.synthesize(verdicts)

        _log_blank(state)

        return {
            "issue_found": result["issue_found"],
            "final_result": result["final_result"],
        }

    return node


def _log_state(state: DebateState, component: str, message: str) -> None:
    if state.get("verbose", False):
        print(f"[{component}] {message}", flush=True)


def _log_blank(state: DebateState) -> None:
    if state.get("verbose", False):
        print(flush=True)


def build_debate_graph(
    client: LLMClient,
    retriever: "LectureNoteRetriever | None" = None,
    roles: list[str] | None = None,
    personas: dict[str, str] | None = None,
    retriever_per_role: dict[str, "LectureNoteRetriever"] | None = None,
    config: GenerationConfig | None = None,
    verbose: bool = False,
    client_per_role: dict[str, LLMClient] | None = None,
):
    roles = roles or ALL_ROLES
    personas = personas or {role: "neutral" for role in roles}
    retriever_per_role = retriever_per_role or {}
    client_per_role = client_per_role or {}

    graph = StateGraph(DebateState)

    for role in roles:
        role_retriever = retriever_per_role.get(role, retriever)
        role_client = client_per_role.get(role, client)

        if role == "conceptual":
            graph.add_node(
                "conceptual_critic",
                make_conceptual_critic_node(
                    role_client,
                    personas.get("conceptual", "neutral"),
                    retriever=role_retriever,
                    config=config,
                ),
            )
        elif role == "procedural":
            graph.add_node(
                "procedural_critic",
                make_procedural_critic_node(
                    role_client,
                    personas.get("procedural", "neutral"),
                    retriever=role_retriever,
                    config=config,
                ),
            )
        elif role == "completeness":
            graph.add_node(
                "completeness_critic",
                make_completeness_critic_node(
                    role_client,
                    role_retriever,
                    config=config,
                ),
            )

    graph.add_node("synthesizer", make_synthesizer_node(roles))

    graph.set_entry_point(f"{roles[0]}_critic")
    for index in range(len(roles) - 1):
        graph.add_edge(f"{roles[index]}_critic", f"{roles[index + 1]}_critic")
    graph.add_edge(f"{roles[-1]}_critic", "synthesizer")
    graph.add_edge("synthesizer", END)

    app = graph.compile()
    app.verbose = verbose
    return app
