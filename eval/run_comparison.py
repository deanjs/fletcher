import time
from pathlib import Path
from typing import TYPE_CHECKING

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

from eval.metrics import evaluate_dataset, print_summary
from fletcher.architectures.debate import build_debate_graph
from fletcher.architectures.persona_debate import run_model_debate, run_persona_debate
from fletcher.architectures.self_critique import SelfCritique, SelfCritique1Pass
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

# K = number of rebuttal rounds allowed once Stage-1 (N-axis, same role)
# critics disagree (README 6.3). K=0 means "evaluate once, never rebut."
# run_persona_debate's max_rounds counts the initial round too, so
# max_rounds = K + 1. Only configs with >=2 personas can ever disagree, so
# N1_* configs are skipped here (K has no effect on a single critic).
K_VALUES = [0, 1, 2, 3]
K_SWEEP_PERSONA_CONFIGS = {
    name: persona_list
    for name, persona_list in PERSONA_CONFIGS.items()
    if len(persona_list) > 1
}

# M-axis: same role, SAME persona (fixed to neutral so persona can't also
# vary), different underlying model. Isolates model diversity from persona
# diversity — see fletcher.architectures.persona_debate.run_model_debate.
# Only roles with a persona-based critic class support this (completeness
# has no persona concept).
M_SWEEP_ROLES = ["conceptual", "procedural"]
M_SWEEP_PERSONA = "neutral"

# Model diversity lives ONLY in the M Sweep (see M_SWEEP_ROLES above), which
# holds persona fixed so model is the sole varying factor. R Sweep, N Sweep,
# and K Sweep intentionally stay single-model (Qwen only) so their results
# aren't confounded by which weights happened to answer — mixing models into
# those would reintroduce exactly the R-sweep-confound problem this project
# already fixed once, just on a different axis. Set to None to skip the M
# Sweep entirely (e.g. when running with a mock/fake backend).
SECONDARY_MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"

# N/K/M axes are studied WITHOUT grounding first, for the same reason they
# stay single-model: mixing in RAG would confound "does persona / debate
# depth / model diversity help" with "does grounding help" — exactly the
# R-sweep-confound problem again, just on a third axis. Once R Sweep shows
# whether RAG matters, add a ("With RAG", top_k) entry here to re-run N/K/M
# with grounding on and check the effect still holds. Each entry is
# (label, top_k_or_None); top_k=None means no retriever is built at all.
NKM_RAG_VARIANTS: list[tuple[str, int | None]] = [("No RAG", None)]
# NKM_RAG_VARIANTS = [("No RAG", None), ("With RAG", 3)]  # <- flip on later


def _flagged_from_critique_text(critique_text: str) -> bool:
    return any(
        keyword in critique_text.lower()
        for keyword in ["incorrect", "wrong", "missing", "error", "inaccurate", "issue"]
    )


def make_arch1a_critic_fn(client):
    """Architecture 1a — literal README 4.1 baseline: ONE critique pass."""
    critic = SelfCritique1Pass(client, verbose=True)

    def critic_fn(explanation: str):
        start = time.perf_counter()
        critique_text = critic.critique(explanation, config=EVAL_CONFIG)
        response = critic.last_response
        latency_ms = (time.perf_counter() - start) * 1000
        flagged = _flagged_from_critique_text(critique_text)
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


def make_arch1b_critic_fn(client):
    """Architecture 1b — adaptive self-critique: critique -> self-review
    ("am I done?") -> revise, looped until sufficient or MAX_ITERATIONS."""
    critic = SelfCritique(client, verbose=True)

    def critic_fn(explanation: str):
        start = time.perf_counter()
        critique_text = critic.critique(explanation, config=EVAL_CONFIG)
        response = critic.last_response
        latency_ms = (time.perf_counter() - start) * 1000
        flagged = _flagged_from_critique_text(critique_text)
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
                "issue_found": False,
                "final_result": "",
                "active_roles": roles,
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


def make_n_sweep_critic_fn(client, persona_list, retriever=None, client_per_key=None, max_rounds=None):
    def critic_fn(explanation: str):
        start = time.perf_counter()
        kwargs = {"max_rounds": max_rounds} if max_rounds is not None else {}
        result = run_persona_debate(
            client,
            persona_list,
            explanation,
            config=EVAL_CONFIG,
            retriever=retriever,
            verbose=True,
            client_per_key=client_per_key,
            **kwargs,
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


def make_m_sweep_critic_fn(role, client_list, retriever=None, max_rounds=None):
    def critic_fn(explanation: str):
        start = time.perf_counter()
        kwargs = {"max_rounds": max_rounds} if max_rounds is not None else {}
        result = run_model_debate(
            role,
            M_SWEEP_PERSONA,
            client_list,
            explanation,
            config=EVAL_CONFIG,
            retriever=retriever,
            verbose=True,
            **kwargs,
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


def run(backend: str = "hf", smoke: bool = False):
    print("Starting FLETCHER evaluation run...", flush=True)
    print(f"Backend: {backend}", flush=True)
    if smoke:
        print("SMOKE TEST MODE: 1 sample/dataset, minimal configs per section.", flush=True)
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

    sample_limit = 1 if smoke else None

    # Smoke mode trims every sweep down to just enough configs to exercise
    # each checklist item cheaply:
    # - one single-critic config (no-debate path) and one multi-critic
    #   config (disagreement-gated round path) for R Sweep
    # - one top_k for the retrieval sweep
    # - one N>=2 persona config (so disagreement/early-stop can be observed)
    # - two K values (off vs on) instead of the full 0-3 range
    # - M Sweep already only has 2 roles, left as-is
    role_configs = ROLE_CONFIGS
    retrieval_top_k_values = RETRIEVAL_TOP_K_VALUES
    persona_configs = PERSONA_CONFIGS
    k_values = K_VALUES
    k_sweep_persona_configs = K_SWEEP_PERSONA_CONFIGS
    if smoke:
        role_configs = {
            "R1_conceptual": ROLE_CONFIGS["R1_conceptual"],
            "R2_conceptual_procedural": ROLE_CONFIGS["R2_conceptual_procedural"],
        }
        retrieval_top_k_values = [3]
        persona_configs = {"N2_strict_merciful": PERSONA_CONFIGS["N2_strict_merciful"]}
        k_values = [0, 2]
        k_sweep_persona_configs = {"N2_strict_merciful": PERSONA_CONFIGS["N2_strict_merciful"]}

    summary_records = []

    print("=== Architecture 1a (Self-Critique, 1-pass baseline) ===", flush=True)
    arch1a_fn = make_arch1a_critic_fn(client)
    print("Running Architecture 1a on Hard Negative dataset...", flush=True)
    arch1a_hn = evaluate_dataset(
        str(HARD_NEGATIVE_PATH),
        arch1a_fn,
        run_label="Architecture 1a / Hard Negative",
        verbose=True,
        limit=sample_limit,
    )
    print_summary(arch1a_hn, label="Hard Negative")
    record_summary(summary_records, "Architecture 1a", "Self-Critique-1Pass", "Hard Negative", arch1a_hn)
    print("Running Architecture 1a on Normal dataset...", flush=True)
    arch1a_normal = evaluate_dataset(
        str(NORMAL_PATH),
        arch1a_fn,
        run_label="Architecture 1a / Normal",
        verbose=True,
        limit=sample_limit,
    )
    print_summary(arch1a_normal, label="Normal")
    record_summary(summary_records, "Architecture 1a", "Self-Critique-1Pass", "Normal", arch1a_normal)

    print("\n=== Architecture 1b (Self-Critique, adaptive) ===", flush=True)
    arch1b_fn = make_arch1b_critic_fn(client)
    print("Running Architecture 1b on Hard Negative dataset...", flush=True)
    arch1b_hn = evaluate_dataset(
        str(HARD_NEGATIVE_PATH),
        arch1b_fn,
        run_label="Architecture 1b / Hard Negative",
        verbose=True,
        limit=sample_limit,
    )
    print_summary(arch1b_hn, label="Hard Negative")
    record_summary(summary_records, "Architecture 1b", "Self-Critique-Adaptive", "Hard Negative", arch1b_hn)
    print("Running Architecture 1b on Normal dataset...", flush=True)
    arch1b_normal = evaluate_dataset(
        str(NORMAL_PATH),
        arch1b_fn,
        run_label="Architecture 1b / Normal",
        verbose=True,
        limit=sample_limit,
    )
    print_summary(arch1b_normal, label="Normal")
    record_summary(summary_records, "Architecture 1b", "Self-Critique-Adaptive", "Normal", arch1b_normal)

    print("\n=== Architecture 2 — R Sweep (No RAG) ===", flush=True)
    for config_name, roles in role_configs.items():
        print(f"\n--- {config_name} ---", flush=True)
        print(f"Building debate graph for roles={roles} without retrieval...", flush=True)
        fn = make_arch2_critic_fn(
            client,
            roles,
            retriever_per_role=None,
        )
        print("Running Hard Negative dataset...", flush=True)
        hn_summary = evaluate_dataset(
            str(HARD_NEGATIVE_PATH),
            fn,
            run_label=f"R Sweep / {config_name} / Hard Negative",
            verbose=True,
            limit=sample_limit,
        )
        print_summary(hn_summary, label="Hard Negative")
        record_summary(summary_records, "R Sweep No RAG", config_name, "Hard Negative", hn_summary)
        print("Running Normal dataset...", flush=True)
        normal_summary = evaluate_dataset(
            str(NORMAL_PATH),
            fn,
            run_label=f"R Sweep / {config_name} / Normal",
            verbose=True,
            limit=sample_limit,
        )
        print_summary(normal_summary, label="Normal")
        record_summary(summary_records, "R Sweep No RAG", config_name, "Normal", normal_summary)

    print("\n=== Architecture 2 — R Sweep (With RAG, top_k=3) ===", flush=True)
    shared_retriever = build_shared_retriever(top_k=3)
    for config_name, roles in role_configs.items():
        print(f"\n--- {config_name} ---", flush=True)
        print(f"Building debate graph for roles={roles} with retrieval...", flush=True)
        fn = make_arch2_critic_fn(
            client,
            roles,
            retriever_per_role=make_retriever_per_role(roles, shared_retriever),
        )
        print("Running Hard Negative dataset...", flush=True)
        hn_summary = evaluate_dataset(
            str(HARD_NEGATIVE_PATH),
            fn,
            run_label=f"R Sweep With RAG / {config_name} / Hard Negative",
            verbose=True,
            limit=sample_limit,
        )
        print_summary(hn_summary, label="Hard Negative")
        record_summary(summary_records, "R Sweep With RAG", config_name, "Hard Negative", hn_summary)
        print("Running Normal dataset...", flush=True)
        normal_summary = evaluate_dataset(
            str(NORMAL_PATH),
            fn,
            run_label=f"R Sweep With RAG / {config_name} / Normal",
            verbose=True,
            limit=sample_limit,
        )
        print_summary(normal_summary, label="Normal")
        record_summary(summary_records, "R Sweep With RAG", config_name, "Normal", normal_summary)

    print("\n=== Architecture 2 — Retrieval Top-K Sweep (R=completeness only) ===", flush=True)
    for top_k in retrieval_top_k_values:
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
            limit=sample_limit,
        )
        print_summary(hn_summary, label="Hard Negative")
        record_summary(summary_records, "Retrieval Top-K", f"top_k={top_k}", "Hard Negative", hn_summary)
        print("Running Normal dataset...", flush=True)
        normal_summary = evaluate_dataset(
            str(NORMAL_PATH),
            fn,
            run_label=f"Retrieval Top-K / top_k={top_k} / Normal",
            verbose=True,
            limit=sample_limit,
        )
        print_summary(normal_summary, label="Normal")
        record_summary(summary_records, "Retrieval Top-K", f"top_k={top_k}", "Normal", normal_summary)

    nkm_retrievers = {
        rag_label: (build_shared_retriever(top_k=top_k) if top_k is not None else None)
        for rag_label, top_k in NKM_RAG_VARIANTS
    }

    for rag_label, retriever in nkm_retrievers.items():
        rag_tag = "" if rag_label == "No RAG" else f" ({rag_label})"

        print(f"\n=== Architecture 2 — N Sweep{rag_tag} (persona-based, R=conceptual only) ===", flush=True)
        for config_name, persona_list in persona_configs.items():
            print(f"\n--- {config_name}{rag_tag} ---", flush=True)
            print(f"Running persona ensemble: {persona_list}", flush=True)
            fn = make_n_sweep_critic_fn(
                client,
                persona_list,
                retriever=retriever,
            )
            print("Running Hard Negative dataset...", flush=True)
            hn_summary = evaluate_dataset(
                str(HARD_NEGATIVE_PATH),
                fn,
                run_label=f"N Sweep{rag_tag} / {config_name} / Hard Negative",
                verbose=True,
                limit=sample_limit,
            )
            print_summary(hn_summary, label="Hard Negative")
            record_summary(summary_records, f"N Sweep{rag_tag}", config_name, "Hard Negative", hn_summary)
            print("Running Normal dataset...", flush=True)
            normal_summary = evaluate_dataset(
                str(NORMAL_PATH),
                fn,
                run_label=f"N Sweep{rag_tag} / {config_name} / Normal",
                verbose=True,
                limit=sample_limit,
            )
            print_summary(normal_summary, label="Normal")
            record_summary(summary_records, f"N Sweep{rag_tag}", config_name, "Normal", normal_summary)

        print(
            f"\n=== Architecture 2 — K Sweep{rag_tag} (Stage 1 debate depth, N>=2 configs only) ===",
            flush=True,
        )
        for k in k_values:
            for config_name, persona_list in k_sweep_persona_configs.items():
                print(f"\n--- K={k} / {config_name}{rag_tag} ---", flush=True)
                fn = make_n_sweep_critic_fn(
                    client,
                    persona_list,
                    retriever=retriever,
                    max_rounds=k + 1,
                )
                print("Running Hard Negative dataset...", flush=True)
                hn_summary = evaluate_dataset(
                    str(HARD_NEGATIVE_PATH),
                    fn,
                    run_label=f"K Sweep{rag_tag} / K={k} / {config_name} / Hard Negative",
                    verbose=True,
                    limit=sample_limit,
                )
                print_summary(hn_summary, label="Hard Negative")
                record_summary(
                    summary_records, f"K Sweep{rag_tag}", f"K={k}/{config_name}", "Hard Negative", hn_summary
                )
                print("Running Normal dataset...", flush=True)
                normal_summary = evaluate_dataset(
                    str(NORMAL_PATH),
                    fn,
                    run_label=f"K Sweep{rag_tag} / K={k} / {config_name} / Normal",
                    verbose=True,
                    limit=sample_limit,
                )
                print_summary(normal_summary, label="Normal")
                record_summary(
                    summary_records, f"K Sweep{rag_tag}", f"K={k}/{config_name}", "Normal", normal_summary
                )

        if secondary_client is None:
            print("\nSkipping M Sweep (no secondary model configured; SECONDARY_MODEL_NAME is None).", flush=True)
        else:
            print(
                f"\n=== Architecture 2 — M Sweep{rag_tag} (model diversity within role, persona fixed=neutral) ===",
                flush=True,
            )
            model_list = [("qwen", client), ("llama", secondary_client)]
            for role in M_SWEEP_ROLES:
                print(f"\n--- {role}{rag_tag} (Qwen + Llama) ---", flush=True)
                fn = make_m_sweep_critic_fn(role, model_list, retriever=retriever)
                print("Running Hard Negative dataset...", flush=True)
                hn_summary = evaluate_dataset(
                    str(HARD_NEGATIVE_PATH),
                    fn,
                    run_label=f"M Sweep{rag_tag} / {role} / Hard Negative",
                    verbose=True,
                    limit=sample_limit,
                )
                print_summary(hn_summary, label="Hard Negative")
                record_summary(summary_records, f"M Sweep{rag_tag}", role, "Hard Negative", hn_summary)
                print("Running Normal dataset...", flush=True)
                normal_summary = evaluate_dataset(
                    str(NORMAL_PATH),
                    fn,
                    run_label=f"M Sweep{rag_tag} / {role} / Normal",
                    verbose=True,
                    limit=sample_limit,
                )
                print_summary(normal_summary, label="Normal")
                record_summary(summary_records, f"M Sweep{rag_tag}", role, "Normal", normal_summary)

    print("\nFLETCHER evaluation run completed.", flush=True)
    print_final_recap(summary_records)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the FLETCHER evaluation sweep.")
    parser.add_argument("--backend", default="hf", help="LLM backend to use (hf, fake).")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: 1 sample/dataset, minimal configs per section. Use to sanity-check "
        "the pipeline before spending GPU time on the full sweep.",
    )
    args = parser.parse_args()
    run(backend=args.backend, smoke=args.smoke)
