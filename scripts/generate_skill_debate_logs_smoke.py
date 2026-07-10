import argparse
from datetime import datetime
from pathlib import Path

from eval.metrics import evaluate_dataset, print_summary
from fletcher.architectures.skill_augmented import run_skill_persona_debate
from fletcher.llm.client import GenerationConfig
from fletcher.llm.factory import create_llm_client
from fletcher.skills import SkillBank, SkillRetriever


DEFAULT_DATASET = Path("eval/datasets/hard_negative/hard_negatives.json")
DEFAULT_OUTPUT_DIR = Path("eval/debate_logs")
DEFAULT_SKILL_BANK = Path("fletcher/skills/skill_bank.jsonl")
DEFAULT_PERSONAS = [("conceptual", "strict"), ("conceptual", "merciful")]
FAKE_JSON_RESPONSE = (
    '{"flagged": true, "confidence": 0.9, '
    '"reasoning": "This explanation contains a known misconception.", '
    '"message_to_others": "This should be flagged."}'
)


def make_client(backend: str):
    if backend == "fake":
        return create_llm_client("fake", canned_response=FAKE_JSON_RESPONSE)
    return create_llm_client(backend)


def run(
    backend: str,
    dataset_path: Path,
    skill_bank_path: Path,
    output_path: Path | None,
    limit: int,
    max_rounds: int,
    top_k: int,
) -> Path:
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = DEFAULT_OUTPUT_DIR / f"{timestamp}_{backend}_minimal_skill_debate.jsonl"

    skill_bank = SkillBank(skill_bank_path)
    if not skill_bank.skills:
        raise ValueError(f"SkillBank is empty or missing: {skill_bank_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("")

    client = make_client(backend)
    skill_retriever = SkillRetriever(skill_bank, top_k=top_k)
    config = GenerationConfig(temperature=0.0, max_new_tokens=256)

    def critic_fn(explanation: str):
        result = run_skill_persona_debate(
            client,
            DEFAULT_PERSONAS,
            explanation,
            skill_retriever=skill_retriever,
            config=config,
            max_rounds=max_rounds,
            verbose=True,
        )
        return (
            result["flagged"],
            0.0,
            result["prompt_tokens"],
            result["completion_tokens"],
            result["llm_calls"],
            result["debate_log"],
        )

    summary = evaluate_dataset(
        str(dataset_path),
        critic_fn,
        run_label=f"Minimal Skill Debate Smoke / {backend}",
        verbose=True,
        limit=limit,
        debate_log_path=output_path,
    )
    print_summary(summary, label="Minimal Skill Debate Smoke")
    print(f"SkillBank loaded from: {skill_bank_path}", flush=True)
    print(f"Skill-augmented debate log written to: {output_path}", flush=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a minimal S=on debate log using a SkillBank.")
    parser.add_argument("--backend", default="hf", help="LLM backend to use: hf or fake.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Dataset JSON path.")
    parser.add_argument("--skill-bank", default=str(DEFAULT_SKILL_BANK), help="SkillBank JSONL path.")
    parser.add_argument("--output", default=None, help="Output JSONL path.")
    parser.add_argument("--limit", type=int, default=1, help="Number of dataset rows to run.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of skills to retrieve per critic call.")
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=2,
        help="Total rounds including the initial evaluation. Use 1 for K=0, 2 for K=1.",
    )
    args = parser.parse_args()

    run(
        backend=args.backend,
        dataset_path=Path(args.dataset),
        skill_bank_path=Path(args.skill_bank),
        output_path=Path(args.output) if args.output else None,
        limit=args.limit,
        max_rounds=args.max_rounds,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
