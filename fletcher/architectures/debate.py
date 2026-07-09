from typing import TYPE_CHECKING, TypedDict

from langgraph.graph import END, StateGraph

from fletcher.agents.content_critic.conceptual import ConceptualCritic
from fletcher.agents.content_critic.procedural import ProceduralCritic
from fletcher.agents.schemas import CriticVerdict
from fletcher.llm.client import GenerationConfig, LLMClient

if TYPE_CHECKING:
    from fletcher.rag.lecture_notes.retriever import LectureNoteRetriever


MAX_ROUNDS = 2
ALL_ROLES = ["conceptual", "procedural", "completeness"]


class DebateState(TypedDict):
    explanation: str
    conceptual_verdict: dict
    procedural_verdict: dict
    completeness_verdict: dict
    conceptual_debate_text: str
    procedural_debate_text: str
    completeness_debate_text: str
    issue_found: bool
    round: int
    final_result: str
    active_roles: list[str]
    debate_history: list[dict]
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
        debate_history = state.get("debate_history", [])
        _log_state(state, "conceptual", "Evaluating the explanation.")
        verdict = critic.evaluate(
            state["explanation"],
            config=config,
            debate_history=debate_history,
        )
        debate_text = critic.compose_debate_turn(
            state["explanation"],
            verdict,
            debate_history=debate_history,
            config=config,
        )
        _log_state(
            state,
            "conceptual",
            f"Verdict flagged={verdict.flagged} confidence={verdict.confidence:.2f}.",
        )
        response = critic.last_response
        debate_response = critic.last_debate_response
        prompt_tokens = 0
        completion_tokens = 0
        llm_calls = 0
        if response:
            prompt_tokens += response.prompt_tokens
            completion_tokens += response.completion_tokens
            llm_calls += 1
        if debate_response:
            prompt_tokens += debate_response.prompt_tokens
            completion_tokens += debate_response.completion_tokens
            llm_calls += 1
        return {
            "conceptual_verdict": verdict.model_dump(),
            "conceptual_debate_text": debate_text,
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
        debate_history = state.get("debate_history", [])
        _log_state(state, "procedural", "Evaluating the explanation.")
        verdict = critic.evaluate(
            state["explanation"],
            config=config,
            debate_history=debate_history,
        )
        debate_text = critic.compose_debate_turn(
            state["explanation"],
            verdict,
            debate_history=debate_history,
            config=config,
        )
        _log_state(
            state,
            "procedural",
            f"Verdict flagged={verdict.flagged} confidence={verdict.confidence:.2f}.",
        )
        response = critic.last_response
        debate_response = critic.last_debate_response
        prompt_tokens = 0
        completion_tokens = 0
        llm_calls = 0
        if response:
            prompt_tokens += response.prompt_tokens
            completion_tokens += response.completion_tokens
            llm_calls += 1
        if debate_response:
            prompt_tokens += debate_response.prompt_tokens
            completion_tokens += debate_response.completion_tokens
            llm_calls += 1
        return {
            "procedural_verdict": verdict.model_dump(),
            "procedural_debate_text": debate_text,
            "total_prompt_tokens": state["total_prompt_tokens"] + prompt_tokens,
            "total_completion_tokens": state["total_completion_tokens"] + completion_tokens,
            "total_llm_calls": state["total_llm_calls"] + llm_calls,
        }

    return node


def make_completeness_critic_node(
    client: LLMClient,
    retriever: "LectureNoteRetriever",
    config: GenerationConfig | None = None,
):
    from fletcher.agents.content_critic.completeness import CompletenessCritic

    critic = CompletenessCritic(client, retriever)

    def node(state: DebateState) -> dict:
        debate_history = state.get("debate_history", [])
        _log_state(state, "completeness", "Evaluating the explanation with retrieval grounding.")
        verdict = critic.evaluate(
            state["explanation"],
            config=config,
            debate_history=debate_history,
        )
        debate_text = critic.compose_debate_turn(
            state["explanation"],
            verdict,
            debate_history=debate_history,
            config=config,
        )
        _log_state(
            state,
            "completeness",
            f"Verdict flagged={verdict.flagged} confidence={verdict.confidence:.2f}.",
        )
        response = critic.last_response
        debate_response = critic.last_debate_response
        prompt_tokens = 0
        completion_tokens = 0
        llm_calls = 0
        if response:
            prompt_tokens += response.prompt_tokens
            completion_tokens += response.completion_tokens
            llm_calls += 1
        if debate_response:
            prompt_tokens += debate_response.prompt_tokens
            completion_tokens += debate_response.completion_tokens
            llm_calls += 1
        return {
            "completeness_verdict": verdict.model_dump(),
            "completeness_debate_text": debate_text,
            "total_prompt_tokens": state["total_prompt_tokens"] + prompt_tokens,
            "total_completion_tokens": state["total_completion_tokens"] + completion_tokens,
            "total_llm_calls": state["total_llm_calls"] + llm_calls,
        }

    return node


def make_orchestrator_node(roles: list[str]):
    def node(state: DebateState) -> dict:
        _log_state(state, "orchestrator", "Checking whether any critic flagged an issue.")
        issue_found = False
        for role in roles:
            verdict_dict = state.get(f"{role}_verdict", {})
            if verdict_dict:
                verdict = CriticVerdict(**verdict_dict)
                if verdict.flagged:
                    issue_found = True
                    break
        _log_state(state, "orchestrator", f"issue_found={issue_found}")
        return {"issue_found": issue_found}

    return node


def debate_round_node(state: DebateState) -> dict:
    next_round = state["round"] + 1
    _log_state(state, "debate_round", f"Archiving round {next_round} and preparing the next round.")
    return {
        "round": next_round,
        "debate_history": [*state.get("debate_history", []), _build_round_entry(state)],
    }


def route_after_orchestrator(state: DebateState) -> str:
    if state["issue_found"] and state["round"] < MAX_ROUNDS:
        return "debate_round"
    return "synthesizer"


def make_synthesizer_node(roles: list[str]):
    def node(state: DebateState) -> dict:
        _log_state(state, "synthesizer", "Combining critic verdicts into the final result.")
        debate_history = state.get("debate_history", [])
        current_round_entry = _build_round_entry(state)
        if not debate_history or debate_history[-1]["round"] != current_round_entry["round"]:
            debate_history = [*debate_history, current_round_entry]

        parts = []
        for role in roles:
            verdict_dict = state.get(f"{role}_verdict", {})
            if verdict_dict:
                verdict = CriticVerdict(**verdict_dict)
                if verdict.flagged:
                    parts.append(f"{role.capitalize()} issue: {verdict.reasoning}")

        if not parts:
            final_result = "No issues found. The explanation is accurate and complete."
        else:
            final_result = " ".join(parts)

        return {
            "final_result": final_result,
            "debate_history": debate_history,
        }

    return node


def _build_round_entry(state: DebateState) -> dict:
    round_number = state["round"] + 1
    verdicts = {}
    debate_turns = {}
    for role in state["active_roles"]:
        verdict_dict = state.get(f"{role}_verdict", {})
        debate_text = state.get(f"{role}_debate_text", "")
        if verdict_dict:
            verdicts[role] = verdict_dict
        if debate_text:
            debate_turns[role] = debate_text

    return {
        "round": round_number,
        "verdicts": verdicts,
        "debate_turns": debate_turns,
    }


def format_debate_trace(result: dict) -> str:
    lines = []
    debate_history = result.get("debate_history", [])

    lines.append("Debate Trace")
    lines.append(f"Rounds: {len(debate_history)}")
    lines.append(f"Final result: {result.get('final_result', '')}")
    lines.append("")

    for round_info in debate_history:
        lines.append(f"=== Round {round_info['round']} ===")
        verdicts = round_info.get("verdicts", {})
        debate_turns = round_info.get("debate_turns", {})

        for role, verdict in verdicts.items():
            role_name = role.replace("_", " ").title()
            lines.append(
                f"[{role_name}] flagged={verdict['flagged']} "
                f"confidence={verdict['confidence']:.2f}"
            )
            lines.append(f"Reasoning: {verdict['reasoning']}")
            if role in debate_turns:
                lines.append(f"Debate text: {debate_turns[role]}")
            lines.append("")

    lines.append(
        "Usage: "
        f"prompt_tokens={result.get('total_prompt_tokens', 0)}, "
        f"completion_tokens={result.get('total_completion_tokens', 0)}, "
        f"llm_calls={result.get('total_llm_calls', 0)}"
    )
    return "\n".join(lines).rstrip()


def print_debate_trace(result: dict) -> None:
    print(format_debate_trace(result))


def _log_state(state: DebateState, component: str, message: str) -> None:
    if state.get("verbose", False):
        current_round = state.get("round", 0) + 1
        print(f"[{component}][round {current_round}] {message}", flush=True)


def build_debate_graph(
    client: LLMClient,
    retriever: "LectureNoteRetriever | None" = None,
    roles: list[str] | None = None,
    personas: dict[str, str] | None = None,
    retriever_per_role: dict[str, "LectureNoteRetriever"] | None = None,
    config: GenerationConfig | None = None,
    verbose: bool = False,
):
    roles = roles or ALL_ROLES
    personas = personas or {role: "neutral" for role in roles}
    retriever_per_role = retriever_per_role or {}

    graph = StateGraph(DebateState)

    for role in roles:
        role_retriever = retriever_per_role.get(role, retriever)

        if role == "conceptual":
            graph.add_node(
                "conceptual_critic",
                make_conceptual_critic_node(
                    client,
                    personas.get("conceptual", "neutral"),
                    retriever=role_retriever,
                    config=config,
                ),
            )
        elif role == "procedural":
            graph.add_node(
                "procedural_critic",
                make_procedural_critic_node(
                    client,
                    personas.get("procedural", "neutral"),
                    retriever=role_retriever,
                    config=config,
                ),
            )
        elif role == "completeness":
            if role_retriever is None:
                raise ValueError("A retriever is required when completeness role is included")
            graph.add_node(
                "completeness_critic",
                make_completeness_critic_node(
                    client,
                    role_retriever,
                    config=config,
                ),
            )

    graph.add_node("orchestrator", make_orchestrator_node(roles))
    graph.add_node("debate_round", debate_round_node)
    graph.add_node("synthesizer", make_synthesizer_node(roles))

    graph.set_entry_point(f"{roles[0]}_critic")
    for index in range(len(roles) - 1):
        graph.add_edge(f"{roles[index]}_critic", f"{roles[index + 1]}_critic")
    graph.add_edge(f"{roles[-1]}_critic", "orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {"debate_round": "debate_round", "synthesizer": "synthesizer"},
    )
    graph.add_edge("debate_round", f"{roles[0]}_critic")
    graph.add_edge("synthesizer", END)

    app = graph.compile()
    app.verbose = verbose
    return app
