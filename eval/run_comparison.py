# import time
# from pathlib import Path

# from fletcher.llm.factory import create_llm_client
# from fletcher.llm.client import GenerationConfig
# from fletcher.architectures.self_critique import SelfCritique
# from fletcher.architectures.debate import build_debate_graph
# from fletcher.agents.content_critic.conceptual import ConceptualCritic
# from eval.metrics import evaluate_dataset, print_summary

# HARD_NEGATIVE_PATH = Path(__file__).parent / "datasets/hard_negative/hard_negatives.json"
# NORMAL_PATH = Path(__file__).parent / "datasets/normal/normal_explanations.json"


# def make_arch1_critic_fn(client):
#     critic = SelfCritique(client)
#     config = GenerationConfig(max_new_tokens=256)

#     def critic_fn(explanation: str):
#         start = time.perf_counter()
#         text = critic.critique(explanation, config=config)
#         latency_ms = (time.perf_counter() - start) * 1000

#         flagged = any(
#             keyword in text.lower()
#             for keyword in ["incorrect", "wrong", "missing", "error", "inaccurate", "issue"]
#         )

#         return flagged, latency_ms, 0, 0

#     return critic_fn


# def make_arch2_critic_fn(client):
#     app = build_debate_graph(client)
#     config = GenerationConfig(max_new_tokens=256)

#     def critic_fn(explanation: str):
#         start = time.perf_counter()
#         result = app.invoke({
#             "explanation": explanation,
#             "conceptual_verdict": {},
#             "procedural_verdict": {},
#             "issue_found": False,
#             "round": 0,
#             "final_result": "",
#         })
#         latency_ms = (time.perf_counter() - start) * 1000

#         flagged = result["issue_found"]

#         return flagged, latency_ms, 0, 0

#     return critic_fn


# def run(backend: str = "hf"):
#     client = create_llm_client(backend)

#     print("=== Architecture 1 (Self-Critique) ===")
#     arch1_fn = make_arch1_critic_fn(client)

#     hn_summary_1 = evaluate_dataset(str(HARD_NEGATIVE_PATH), arch1_fn)
#     print_summary(hn_summary_1, label="Hard Negative")

#     normal_summary_1 = evaluate_dataset(str(NORMAL_PATH), arch1_fn)
#     print_summary(normal_summary_1, label="Normal")

#     print("\n=== Architecture 2 (Debate) ===")
#     arch2_fn = make_arch2_critic_fn(client)

#     hn_summary_2 = evaluate_dataset(str(HARD_NEGATIVE_PATH), arch2_fn)
#     print_summary(hn_summary_2, label="Hard Negative")

#     normal_summary_2 = evaluate_dataset(str(NORMAL_PATH), arch2_fn)
#     print_summary(normal_summary_2, label="Normal")


# if __name__ == "__main__":
#     run()

import time
from pathlib import Path

from fletcher.llm.factory import create_llm_client
from fletcher.llm.client import GenerationConfig
from fletcher.architectures.self_critique import SelfCritique
from fletcher.architectures.debate import build_debate_graph
from fletcher.rag.lecture_notes.retriever import LectureNoteRetriever
from eval.metrics import evaluate_dataset, print_summary

HARD_NEGATIVE_PATH = Path(__file__).parent / "datasets/hard_negative/hard_negatives.json"
NORMAL_PATH = Path(__file__).parent / "datasets/normal/normal_explanations.json"


def make_arch1_critic_fn(client):
    critic = SelfCritique(client)
    config = GenerationConfig(max_new_tokens=256)

    def critic_fn(explanation: str):
        start = time.perf_counter()
        text = critic.critique(explanation, config=config)
        latency_ms = (time.perf_counter() - start) * 1000

        flagged = any(
            keyword in text.lower()
            for keyword in ["incorrect", "wrong", "missing", "error", "inaccurate", "issue"]
        )

        return flagged, latency_ms, 0, 0

    return critic_fn


def make_arch2_critic_fn(client, retriever):
    app = build_debate_graph(client, retriever)
    config = GenerationConfig(max_new_tokens=256)

    def critic_fn(explanation: str):
        start = time.perf_counter()
        result = app.invoke({
            "explanation": explanation,
            "conceptual_verdict": {},
            "procedural_verdict": {},
            "completeness_verdict": {},
            "issue_found": False,
            "round": 0,
            "final_result": "",
        })
        latency_ms = (time.perf_counter() - start) * 1000

        flagged = result["issue_found"]

        return flagged, latency_ms, 0, 0

    return critic_fn


def run(backend: str = "hf"):
    client = create_llm_client(backend)

    retriever = LectureNoteRetriever()
    retriever.build_index()

    print("=== Architecture 1 (Self-Critique) ===")
    arch1_fn = make_arch1_critic_fn(client)

    hn_summary_1 = evaluate_dataset(str(HARD_NEGATIVE_PATH), arch1_fn)
    print_summary(hn_summary_1, label="Hard Negative")

    normal_summary_1 = evaluate_dataset(str(NORMAL_PATH), arch1_fn)
    print_summary(normal_summary_1, label="Normal")

    print("\n=== Architecture 2 (Debate, 3 Critics) ===")
    arch2_fn = make_arch2_critic_fn(client, retriever)

    hn_summary_2 = evaluate_dataset(str(HARD_NEGATIVE_PATH), arch2_fn)
    print_summary(hn_summary_2, label="Hard Negative")

    normal_summary_2 = evaluate_dataset(str(NORMAL_PATH), arch2_fn)
    print_summary(normal_summary_2, label="Normal")


if __name__ == "__main__":
    run()