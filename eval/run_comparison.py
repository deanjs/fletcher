import time
from pathlib import Path
from typing import TYPE_CHECKING

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

from eval.metrics import evaluate_dataset, print_summary
from fletcher.architectures.debate import build_debate_graph
from fletcher.architectures.persona_debate import run_persona_debate
from fletcher.architectures.self_critique import SelfCritique
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

# Architecture 2 only (never Architecture 1 — self-critique is defined as the
# SAME model checking itself, so mixing models there would break the baseline).
# Giving one critic/persona a different, smaller model is a first step toward
# reducing the "critics agree because they're the same weights" homogeneity
# problem in debate. Kept small (one role, one persona) to fit in a single
# Colab GPU alongside the main 7B model; set to None to disable and fall back
# to the single-model behavior.
SECONDARY_MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
SECONDARY_MODEL_ROLE = "completeness"
SECONDARY_MODEL_PERSONA_KEY = "conceptual_merciful"


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
            critic.last_llm_calls,
        )

    return critic_fn


def make_arch2_critic_fn(client, roles, retriever_per_role=None, client_per_role=None):
    app = build_debate_graph(
        client,
        roles=roles,
        retriever_per_role=retriever_per_role,
        config=EVAL_CONFIG,
        verbose=True,
        client_per_role=client_per_role,
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
                "verbose": True,
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


def make_n_sweep_critic_fn(client, persona_list, retriever=None, client_per_key=None):
    def critic_fn(explanation: str):
        start = time.perf_counter()
        result = run_persona_debate(
            client,
            persona_list,
            explanation,
            config=EVAL_CONFIG,
            retriever=retriever,
            verbose=True,
            client_per_key=client_per_key,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        return (
            result["flagged"],
            latency_ms,
            result["prompt_tokens"],
            result["completion_tokens"],
            result["llm_calls"],
        )

    return critic_fn


def build_shared_retriever(top_k: int = 3) -> "LectureNoteRetriever":
    from fletcher.rag.lecture_notes.retriever import LectureNoteRetriever

    print(f"Preparing shared retriever with top_k={top_k}...", flush=True)
    retriever = LectureNoteRetriever(top_k=top_k)
    retriever.build_index()
    print(f"Shared retriever ready with top_k={top_k}.", flush=True)
    return retriever


def make_retriever_per_role(roles, retriever):
    return {role: retriever for role in roles}


def make_client_per_role(roles, secondary_client):
    if secondary_client is None or SECONDARY_MODEL_ROLE not in roles:
        return None
    return {SECONDARY_MODEL_ROLE: secondary_client}


def make_client_per_key(persona_list, secondary_client):
    if secondary_client is None:
        return None
    keys = {f"{role}_{persona}" for role, persona in persona_list}
    if SECONDARY_MODEL_PERSONA_KEY not in keys:
        return None
    return {SECONDARY_MODEL_PERSONA_KEY: secondary_client}


def record_summary(summary_records, section: str, config_name: str, dataset_label: str, summary) -> None:
    summary_records.append({
        "section": section,
        "config": config_name,
        "dataset": dataset_label,
        "accuracy": summary.accuracy,
        "latency_ms": summary.avg_latency_ms,
        "llm_calls": summary.avg_llm_calls,
    })


def print_final_recap(summary_records: list[dict]) -> None:
    print("\n=== Final Experiment Recap ===", flush=True)
    for record in summary_records:
        print(
            f"[{record['section']}] {record['config']} / {record['dataset']} "
            f"accuracy={record['accuracy']:.2%} "
            f"avg_latency_ms={record['latency_ms']:.1f} "
            f"avg_llm_calls={record['llm_calls']:.1f}",
            flush=True,
        )


def run(backend: str = "hf"):
    print("Starting FLETCHER evaluation run...", flush=True)
    print(f"Backend: {backend}", flush=True)
    if torch is not None:
        torch.manual_seed(42)
        print("Torch manual seed set to 42.", flush=True)
    else:
        print("Torch is not installed in this environment.", flush=True)

    print("Initializing LLM client...", flush=True)
    client = create_llm_client(backend)
    print("LLM client initialized.", flush=True)

    secondary_client = None
    if backend == "hf" and SECONDARY_MODEL_NAME:
        print(f"Initializing secondary LLM client ({SECONDARY_MODEL_NAME}) for Architecture 2 model diversity...", flush=True)
        secondary_client = create_llm_client(backend, model_name=SECONDARY_MODEL_NAME)
        print("Secondary LLM client initialized.", flush=True)

    summary_records = []

    print("=== Architecture 1 (Self-Critique) ===", flush=True)
    arch1_fn = make_arch1_critic_fn(client)
    print("Running Architecture 1 on Hard Negative dataset...", flush=True)
    arch1_hn = evaluate_dataset(
        str(HARD_NEGATIVE_PATH),
        arch1_fn,
        run_label="Architecture 1 / Hard Negative",
        verbose=True,
    )
    print_summary(arch1_hn, label="Hard Negative")
    record_summary(summary_records, "Architecture 1", "Self-Critique", "Hard Negative", arch1_hn)
    print("Running Architecture 1 on Normal dataset...", flush=True)
    arch1_normal = evaluate_dataset(
        str(NORMAL_PATH),
        arch1_fn,
        run_label="Architecture 1 / Normal",
        verbose=True,
    )
    print_summary(arch1_normal, label="Normal")
    record_summary(summary_records, "Architecture 1", "Self-Critique", "Normal", arch1_normal)

    print("\n=== Architecture 2 — R Sweep (No RAG) ===", flush=True)
    for config_name, roles in ROLE_CONFIGS.items():
        print(f"\n--- {config_name} ---", flush=True)
        print(f"Building debate graph for roles={roles} without retrieval...", flush=True)
        fn = make_arch2_critic_fn(
            client,
            roles,
            retriever_per_role=None,
            client_per_role=make_client_per_role(roles, secondary_client),
        )
        print("Running Hard Negative dataset...", flush=True)
        hn_summary = evaluate_dataset(
            str(HARD_NEGATIVE_PATH),
            fn,
            run_label=f"R Sweep / {config_name} / Hard Negative",
            verbose=True,
        )
        print_summary(hn_summary, label="Hard Negative")
        record_summary(summary_records, "R Sweep No RAG", config_name, "Hard Negative", hn_summary)
        print("Running Normal dataset...", flush=True)
        normal_summary = evaluate_dataset(
            str(NORMAL_PATH),
            fn,
            run_label=f"R Sweep / {config_name} / Normal",
            verbose=True,
        )
        print_summary(normal_summary, label="Normal")
        record_summary(summary_records, "R Sweep No RAG", config_name, "Normal", normal_summary)

    print("\n=== Architecture 2 — R Sweep (With RAG, top_k=3) ===", flush=True)
    shared_retriever = build_shared_retriever(top_k=3)
    for config_name, roles in ROLE_CONFIGS.items():
        print(f"\n--- {config_name} ---", flush=True)
        print(f"Building debate graph for roles={roles} with retrieval...", flush=True)
        fn = make_arch2_critic_fn(
            client,
            roles,
            retriever_per_role=make_retriever_per_role(roles, shared_retriever),
            client_per_role=make_client_per_role(roles, secondary_client),
        )
        print("Running Hard Negative dataset...", flush=True)
        hn_summary = evaluate_dataset(
            str(HARD_NEGATIVE_PATH),
            fn,
            run_label=f"R Sweep With RAG / {config_name} / Hard Negative",
            verbose=True,
        )
        print_summary(hn_summary, label="Hard Negative")
        record_summary(summary_records, "R Sweep With RAG", config_name, "Hard Negative", hn_summary)
        print("Running Normal dataset...", flush=True)
        normal_summary = evaluate_dataset(
            str(NORMAL_PATH),
            fn,
            run_label=f"R Sweep With RAG / {config_name} / Normal",
            verbose=True,
        )
        print_summary(normal_summary, label="Normal")
        record_summary(summary_records, "R Sweep With RAG", config_name, "Normal", normal_summary)

    print("\n=== Architecture 2 — Retrieval Top-K Sweep (R=completeness only) ===", flush=True)
    for top_k in RETRIEVAL_TOP_K_VALUES:
        print(f"\n--- top_k={top_k} ---", flush=True)
        retriever = build_shared_retriever(top_k=top_k)
        print("Building completeness-only debate graph...", flush=True)
        fn = make_arch2_critic_fn(
            client,
            ["completeness"],
            retriever_per_role={"completeness": retriever},
        )
        print("Running Hard Negative dataset...", flush=True)
        hn_summary = evaluate_dataset(
            str(HARD_NEGATIVE_PATH),
            fn,
            run_label=f"Retrieval Top-K / top_k={top_k} / Hard Negative",
            verbose=True,
        )
        print_summary(hn_summary, label="Hard Negative")
        record_summary(summary_records, "Retrieval Top-K", f"top_k={top_k}", "Hard Negative", hn_summary)
        print("Running Normal dataset...", flush=True)
        normal_summary = evaluate_dataset(
            str(NORMAL_PATH),
            fn,
            run_label=f"Retrieval Top-K / top_k={top_k} / Normal",
            verbose=True,
        )
        print_summary(normal_summary, label="Normal")
        record_summary(summary_records, "Retrieval Top-K", f"top_k={top_k}", "Normal", normal_summary)

    print("\n=== Architecture 2 — N Sweep (persona-based, R=conceptual only) ===", flush=True)
    for config_name, persona_list in PERSONA_CONFIGS.items():
        print(f"\n--- {config_name} ---", flush=True)
        print(f"Running persona ensemble: {persona_list}", flush=True)
        fn = make_n_sweep_critic_fn(
            client,
            persona_list,
            client_per_key=make_client_per_key(persona_list, secondary_client),
        )
        print("Running Hard Negative dataset...", flush=True)
        hn_summary = evaluate_dataset(
            str(HARD_NEGATIVE_PATH),
            fn,
            run_label=f"N Sweep / {config_name} / Hard Negative",
            verbose=True,
        )
        print_summary(hn_summary, label="Hard Negative")
        record_summary(summary_records, "N Sweep", config_name, "Hard Negative", hn_summary)
        print("Running Normal dataset...", flush=True)
        normal_summary = evaluate_dataset(
            str(NORMAL_PATH),
            fn,
            run_label=f"N Sweep / {config_name} / Normal",
            verbose=True,
        )
        print_summary(normal_summary, label="Normal")
        record_summary(summary_records, "N Sweep", config_name, "Normal", normal_summary)

    print("\nFLETCHER evaluation run completed.", flush=True)
    print_final_recap(summary_records)


if __name__ == "__main__":
    run()
