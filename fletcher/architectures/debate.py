# from typing import TypedDict
# from langgraph.graph import StateGraph, END

# from fletcher.agents.schemas import CriticVerdict
# from fletcher.agents.content_critic.conceptual import ConceptualCritic
# from fletcher.agents.content_critic.procedural import ProceduralCritic
# from fletcher.llm.client import LLMClient

# MAX_ROUNDS = 2


# class DebateState(TypedDict):
#     explanation: str
#     conceptual_verdict: dict
#     procedural_verdict: dict
#     issue_found: bool
#     round: int
#     final_result: str


# def make_conceptual_critic_node(client: LLMClient):
#     critic = ConceptualCritic(client)

#     def node(state: DebateState) -> dict:
#         verdict = critic.evaluate(state["explanation"])
#         return {"conceptual_verdict": verdict.model_dump()}

#     return node


# def make_procedural_critic_node(client: LLMClient):
#     critic = ProceduralCritic(client)

#     def node(state: DebateState) -> dict:
#         verdict = critic.evaluate(state["explanation"])
#         return {"procedural_verdict": verdict.model_dump()}

#     return node


# def orchestrator_node(state: DebateState) -> dict:
#     conceptual = CriticVerdict(**state["conceptual_verdict"])
#     procedural = CriticVerdict(**state["procedural_verdict"])

#     issue_found = conceptual.flagged or procedural.flagged

#     return {"issue_found": issue_found}


# def debate_round_node(state: DebateState) -> dict:
#     return {"round": state["round"] + 1}


# def route_after_orchestrator(state: DebateState) -> str:
#     if state["issue_found"] and state["round"] < MAX_ROUNDS:
#         return "debate_round"
#     return "synthesizer"


# def synthesizer_node(state: DebateState) -> dict:
#     conceptual = CriticVerdict(**state["conceptual_verdict"])
#     procedural = CriticVerdict(**state["procedural_verdict"])

#     parts = []
#     if conceptual.flagged:
#         parts.append(f"Conceptual issue: {conceptual.reasoning}")
#     if procedural.flagged:
#         parts.append(f"Procedural issue: {procedural.reasoning}")

#     if not parts:
#         final_result = "No issues found. The explanation is accurate."
#     else:
#         final_result = " ".join(parts)

#     return {"final_result": final_result}


# def build_debate_graph(client: LLMClient):
#     graph = StateGraph(DebateState)

#     graph.add_node("conceptual_critic", make_conceptual_critic_node(client))
#     graph.add_node("procedural_critic", make_procedural_critic_node(client))
#     graph.add_node("orchestrator", orchestrator_node)
#     graph.add_node("debate_round", debate_round_node)
#     graph.add_node("synthesizer", synthesizer_node)

#     graph.set_entry_point("conceptual_critic")
#     graph.add_edge("conceptual_critic", "procedural_critic")
#     graph.add_edge("procedural_critic", "orchestrator")

#     graph.add_conditional_edges(
#         "orchestrator",
#         route_after_orchestrator,
#         {"debate_round": "debate_round", "synthesizer": "synthesizer"},
#     )

#     graph.add_edge("debate_round", "conceptual_critic")
#     graph.add_edge("synthesizer", END)

#     return graph.compile()

# from typing import TypedDict
# from langgraph.graph import StateGraph, END

# from fletcher.agents.schemas import CriticVerdict
# from fletcher.agents.content_critic.conceptual import ConceptualCritic
# from fletcher.agents.content_critic.procedural import ProceduralCritic
# from fletcher.agents.content_critic.completeness import CompletenessCritic
# from fletcher.llm.client import LLMClient
# from fletcher.rag.lecture_notes.retriever import LectureNoteRetriever

# MAX_ROUNDS = 2


# class DebateState(TypedDict):
#     explanation: str
#     conceptual_verdict: dict
#     procedural_verdict: dict
#     completeness_verdict: dict
#     issue_found: bool
#     round: int
#     final_result: str


# def make_conceptual_critic_node(client: LLMClient):
#     critic = ConceptualCritic(client)

#     def node(state: DebateState) -> dict:
#         verdict = critic.evaluate(state["explanation"])
#         return {"conceptual_verdict": verdict.model_dump()}

#     return node


# def make_procedural_critic_node(client: LLMClient):
#     critic = ProceduralCritic(client)

#     def node(state: DebateState) -> dict:
#         verdict = critic.evaluate(state["explanation"])
#         return {"procedural_verdict": verdict.model_dump()}

#     return node


# def make_completeness_critic_node(client: LLMClient, retriever: LectureNoteRetriever):
#     critic = CompletenessCritic(client, retriever)

#     def node(state: DebateState) -> dict:
#         verdict = critic.evaluate(state["explanation"])
#         return {"completeness_verdict": verdict.model_dump()}

#     return node


# def orchestrator_node(state: DebateState) -> dict:
#     conceptual = CriticVerdict(**state["conceptual_verdict"])
#     procedural = CriticVerdict(**state["procedural_verdict"])
#     completeness = CriticVerdict(**state["completeness_verdict"])

#     issue_found = conceptual.flagged or procedural.flagged or completeness.flagged

#     return {"issue_found": issue_found}


# def debate_round_node(state: DebateState) -> dict:
#     return {"round": state["round"] + 1}


# def route_after_orchestrator(state: DebateState) -> str:
#     if state["issue_found"] and state["round"] < MAX_ROUNDS:
#         return "debate_round"
#     return "synthesizer"


# def synthesizer_node(state: DebateState) -> dict:
#     conceptual = CriticVerdict(**state["conceptual_verdict"])
#     procedural = CriticVerdict(**state["procedural_verdict"])
#     completeness = CriticVerdict(**state["completeness_verdict"])

#     parts = []
#     if conceptual.flagged:
#         parts.append(f"Conceptual issue: {conceptual.reasoning}")
#     if procedural.flagged:
#         parts.append(f"Procedural issue: {procedural.reasoning}")
#     if completeness.flagged:
#         parts.append(f"Completeness issue: {completeness.reasoning}")

#     if not parts:
#         final_result = "No issues found. The explanation is accurate and complete."
#     else:
#         final_result = " ".join(parts)

#     return {"final_result": final_result}


# def build_debate_graph(client: LLMClient, retriever: LectureNoteRetriever):
#     graph = StateGraph(DebateState)

#     graph.add_node("conceptual_critic", make_conceptual_critic_node(client))
#     graph.add_node("procedural_critic", make_procedural_critic_node(client))
#     graph.add_node("completeness_critic", make_completeness_critic_node(client, retriever))
#     graph.add_node("orchestrator", orchestrator_node)
#     graph.add_node("debate_round", debate_round_node)
#     graph.add_node("synthesizer", synthesizer_node)

#     graph.set_entry_point("conceptual_critic")
#     graph.add_edge("conceptual_critic", "procedural_critic")
#     graph.add_edge("procedural_critic", "completeness_critic")
#     graph.add_edge("completeness_critic", "orchestrator")

#     graph.add_conditional_edges(
#         "orchestrator",
#         route_after_orchestrator,
#         {"debate_round": "debate_round", "synthesizer": "synthesizer"},
#     )

#     graph.add_edge("debate_round", "conceptual_critic")
#     graph.add_edge("synthesizer", END)

#     return graph.compile()

from typing import TypedDict
from langgraph.graph import StateGraph, END

from fletcher.agents.schemas import CriticVerdict
from fletcher.agents.content_critic.conceptual import ConceptualCritic
from fletcher.agents.content_critic.procedural import ProceduralCritic
from fletcher.agents.content_critic.completeness import CompletenessCritic
from fletcher.llm.client import LLMClient
from fletcher.rag.lecture_notes.retriever import LectureNoteRetriever

MAX_ROUNDS = 2
ALL_ROLES = ["conceptual", "procedural", "completeness"]


class DebateState(TypedDict):
    explanation: str
    conceptual_verdict: dict
    procedural_verdict: dict
    completeness_verdict: dict
    issue_found: bool
    round: int
    final_result: str
    active_roles: list


def make_conceptual_critic_node(client: LLMClient, persona: str = "neutral"):
    critic = ConceptualCritic(client, persona=persona)

    def node(state: DebateState) -> dict:
        verdict = critic.evaluate(state["explanation"])
        return {"conceptual_verdict": verdict.model_dump()}

    return node


def make_procedural_critic_node(client: LLMClient, persona: str = "neutral"):
    critic = ProceduralCritic(client, persona=persona)

    def node(state: DebateState) -> dict:
        verdict = critic.evaluate(state["explanation"])
        return {"procedural_verdict": verdict.model_dump()}

    return node


def make_completeness_critic_node(client: LLMClient, retriever: LectureNoteRetriever):
    critic = CompletenessCritic(client, retriever)

    def node(state: DebateState) -> dict:
        verdict = critic.evaluate(state["explanation"])
        return {"completeness_verdict": verdict.model_dump()}

    return node


def make_orchestrator_node(roles: list[str]):
    def node(state: DebateState) -> dict:
        issue_found = False
        for role in roles:
            verdict_dict = state.get(f"{role}_verdict", {})
            if verdict_dict:
                verdict = CriticVerdict(**verdict_dict)
                if verdict.flagged:
                    issue_found = True
                    break
        return {"issue_found": issue_found}
    return node


def debate_round_node(state: DebateState) -> dict:
    return {"round": state["round"] + 1}


def route_after_orchestrator(state: DebateState) -> str:
    if state["issue_found"] and state["round"] < MAX_ROUNDS:
        return "debate_round"
    return "synthesizer"


def make_synthesizer_node(roles: list[str]):
    def node(state: DebateState) -> dict:
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

        return {"final_result": final_result}
    return node


def build_debate_graph(
    client: LLMClient,
    retriever: LectureNoteRetriever | None = None,
    roles: list[str] | None = None,
    personas: dict[str, str] | None = None,
):
    roles = roles or ALL_ROLES
    personas = personas or {role: "neutral" for role in roles}

    if "completeness" in roles:
        assert retriever is not None, "retriever is required when completeness role is included"

    graph = StateGraph(DebateState)

    # add critic nodes
    for role in roles:
        if role == "conceptual":
            graph.add_node("conceptual_critic", make_conceptual_critic_node(client, personas.get("conceptual", "neutral")))
        elif role == "procedural":
            graph.add_node("procedural_critic", make_procedural_critic_node(client, personas.get("procedural", "neutral")))
        elif role == "completeness":
            graph.add_node("completeness_critic", make_completeness_critic_node(client, retriever))

    graph.add_node("orchestrator", make_orchestrator_node(roles))
    graph.add_node("debate_round", debate_round_node)
    graph.add_node("synthesizer", make_synthesizer_node(roles))

    # chain critics sequentially
    graph.set_entry_point(f"{roles[0]}_critic")
    for i in range(len(roles) - 1):
        graph.add_edge(f"{roles[i]}_critic", f"{roles[i+1]}_critic")
    graph.add_edge(f"{roles[-1]}_critic", "orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {"debate_round": "debate_round", "synthesizer": "synthesizer"},
    )
    graph.add_edge("debate_round", f"{roles[0]}_critic")
    graph.add_edge("synthesizer", END)

    return graph.compile()