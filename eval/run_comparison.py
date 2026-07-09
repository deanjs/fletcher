import time
from pathlib import Path
from typing import TYPE_CHECKING

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

from eval.metrics import evaluate_dataset, print_summary
from fletcher.architectures.debate import build_debate_graph
from fletcher.architectures.self_critique import SelfCritique
from fletcher.agents.content_critic.conceptual import ConceptualCritic
from fletcher.agents.content_critic.procedural import ProceduralCritic
from fletcher.llm.client import GenerationConfig
from fletcher.llm.factory import create_llm_client

if TYPE_CHECKING:
    from fletcher.rag.lecture_notes.retriever import LectureNoteRetriever


HARD_NEGATIVE_PATH = Path(__file__).parent / "datasets/hard_negative/hard_negatives.json"
NORMAL_PATH = Path(__file__).parent / "datasets/normal/normal_explanations.json"
EVAL_CONFIG = GenerationConfig(temperature=0.0, max_new_tokens=256)

ROLE_CONFIGS = {
    "R1_conceptual": ["conceptual"],
    "R1_procedural": ["procedural"],
    "R1_completeness": ["completeness"],
    "R2_conceptual_procedural": ["conceptual", "procedural"],
    "R2_conceptual_completeness": ["conceptual", "completeness"],
    "R3_all": ["conceptual", "procedural", "completeness"],
}

PERSONA_CONFIGS = {
    "N1_neutral": [("conceptual", "neutral")],
    "N1_strict": [("conceptual", "strict")],
    "N1_merciful": [("conceptual", "merciful")],
    "N2_strict_merciful": [("conceptual", "strict"), ("conceptual", "merciful")],
    "N3_all_personas": [
        ("conceptual", "strict"),
        ("conceptual", "neutral"),
        ("conceptual", "merciful"),
    ],
}

RETRIEVAL_TOP_K_VALUES = [1, 3, 5]


def make_arch1_critic_fn(client):
    critic = SelfCritique(client)

    def critic_fn(explanation: str):
        start = time.perf_counter()
        critique_text = critic.critique(explanation, config=EVAL_CONFIG)
        response = critic.last_response
        latency_ms = (time.perf_counter() - start) * 1000
        flagged = any(
            keyword in critique_text.lower()
            for keyword in ["incorrect", "wrong", "missing", "error", "inaccurate", "issue"]
        )
        if response is None:
            return flagged, latency_ms, 0, 0, 0
        return (
            flagged,
            latency_ms,
            response.prompt_tokens,
            response.completion_tokens,
            3,
        )

    return critic_fn


def make_arch2_critic_fn(client, roles, retriever_per_role=None):
    app = build_debate_graph(
        client,
        roles=roles,
        retriever_per_role=retriever_per_role,
        config=EVAL_CONFIG,
    )

    def critic_fn(explanation: str):
        start = time.perf_counter()
        result = app.invoke(
            {
                "explanation": explanation,
                "conceptual_verdict": {},
                "procedural_verdict": {},
                "completeness_verdict": {},
                "conceptual_debate_text": "",
                "procedural_debate_text": "",
                "completeness_debate_text": "",
                "issue_found": False,
                "round": 0,
                "final_result": "",
                "active_roles": roles,
                "debate_history": [],
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_llm_calls": 0,
            }
        )
        latency_ms = (time.perf_counter() - start) * 1000
        return (
            result["issue_found"],
            latency_ms,
            result["total_prompt_tokens"],
            result["total_completion_tokens"],
            result["total_llm_calls"],
        )

    return critic_fn


def make_n_sweep_critic_fn(client, persona_list, retriever=None):
    def critic_fn(explanation: str):
        votes = []
        total_latency = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_llm_calls = 0

        for role, persona in persona_list:
            if role == "conceptual":
                critic = ConceptualCritic(client, persona=persona, retriever=retriever)
            elif role == "procedural":
                critic = ProceduralCritic(client, persona=persona, retriever=retriever)
            else:
                raise ValueError(f"Unsupported role for persona sweep: {role}")

            start = time.perf_counter()
            verdict = critic.evaluate(explanation, config=EVAL_CONFIG)
            total_latency += (time.perf_counter() - start) * 1000
            response = critic.last_response
            if response is not None:
                total_prompt_tokens += response.prompt_tokens
                total_completion_tokens += response.completion_tokens
                total_llm_calls += 1
            votes.append(verdict.flagged)

        flagged_count = sum(votes)
        not_flagged_count = len(votes) - flagged_count
        flagged = flagged_count > not_flagged_count
        return (
            flagged,
            total_latency,
            total_prompt_tokens,
            total_completion_tokens,
            total_llm_calls,
        )

    return critic_fn


def build_shared_retriever(top_k: int = 3) -> "LectureNoteRetriever":
    from fletcher.rag.lecture_notes.retriever import LectureNoteRetriever

    retriever = LectureNoteRetriever(top_k=top_k)
    retriever.build_index()
    return retriever


def make_retriever_per_role(roles, retriever):
    return {role: retriever for role in roles}


def run(backend: str = "hf"):
    if torch is not None:
        torch.manual_seed(42)
    client = create_llm_client(backend)

    print("=== Architecture 1 (Self-Critique) ===")
    arch1_fn = make_arch1_critic_fn(client)
    print_summary(evaluate_dataset(str(HARD_NEGATIVE_PATH), arch1_fn), label="Hard Negative")
    print_summary(evaluate_dataset(str(NORMAL_PATH), arch1_fn), label="Normal")

    print("\n=== Architecture 2 — R Sweep (No RAG) ===")
    for config_name, roles in ROLE_CONFIGS.items():
        print(f"\n--- {config_name} ---")
        fn = make_arch2_critic_fn(client, roles, retriever_per_role=None)
        print_summary(evaluate_dataset(str(HARD_NEGATIVE_PATH), fn), label="Hard Negative")
        print_summary(evaluate_dataset(str(NORMAL_PATH), fn), label="Normal")

    print("\n=== Architecture 2 — R Sweep (With RAG, top_k=3) ===")
    shared_retriever = build_shared_retriever(top_k=3)
    for config_name, roles in ROLE_CONFIGS.items():
        print(f"\n--- {config_name} ---")
        fn = make_arch2_critic_fn(
            client,
            roles,
            retriever_per_role=make_retriever_per_role(roles, shared_retriever),
        )
        print_summary(evaluate_dataset(str(HARD_NEGATIVE_PATH), fn), label="Hard Negative")
        print_summary(evaluate_dataset(str(NORMAL_PATH), fn), label="Normal")

    print("\n=== Architecture 2 — Retrieval Top-K Sweep (R=completeness only) ===")
    for top_k in RETRIEVAL_TOP_K_VALUES:
        print(f"\n--- top_k={top_k} ---")
        retriever = build_shared_retriever(top_k=top_k)
        fn = make_arch2_critic_fn(
            client,
            ["completeness"],
            retriever_per_role={"completeness": retriever},
        )
        print_summary(evaluate_dataset(str(HARD_NEGATIVE_PATH), fn), label="Hard Negative")
        print_summary(evaluate_dataset(str(NORMAL_PATH), fn), label="Normal")

    print("\n=== Architecture 2 — N Sweep (persona-based, R=conceptual only) ===")
    for config_name, persona_list in PERSONA_CONFIGS.items():
        print(f"\n--- {config_name} ---")
        fn = make_n_sweep_critic_fn(client, persona_list)
        print_summary(evaluate_dataset(str(HARD_NEGATIVE_PATH), fn), label="Hard Negative")
        print_summary(evaluate_dataset(str(NORMAL_PATH), fn), label="Normal")


if __name__ == "__main__":
    run()
