import argparse
from datetime import datetime
from pathlib import Path

from eval.metrics import evaluate_dataset, print_summary
from fletcher.architectures.persona_debate import run_persona_debate
from fletcher.llm.client import GenerationConfig
from fletcher.llm.factory import create_llm_client


DEFAULT_DATASET = Path("eval/datasets/hard_negative/hard_negatives.json")
DEFAULT_OUTPUT_DIR = Path("eval/debate_logs")
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
    output_path: Path | None,
    limit: int,
    max_rounds: int,
) -> Path:
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = DEFAULT_OUTPUT_DIR / f"{timestamp}_{backend}_minimal_debate.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("")

    client = make_client(backend)
    config = GenerationConfig(temperature=0.0, max_new_tokens=256)

    def critic_fn(explanation: str):
        result = run_persona_debate(
            client,
            DEFAULT_PERSONAS,
            explanation,
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
        run_label=f"Minimal Debate Smoke / {backend}",
        verbose=True,
        limit=limit,
        debate_log_path=output_path,
    )
    print_summary(summary, label="Minimal Debate Smoke")
    print(f"Debate log written to: {output_path}", flush=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a minimal Architecture 2 debate log for SkillBank distillation.")
    parser.add_argument("--backend", default="hf", help="LLM backend to use: hf or fake.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Dataset JSON path.")
    parser.add_argument("--output", default=None, help="Output JSONL path.")
    parser.add_argument("--limit", type=int, default=1, help="Number of dataset rows to run.")
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
        output_path=Path(args.output) if args.output else None,
        limit=args.limit,
        max_rounds=args.max_rounds,
    )


if __name__ == "__main__":
    main()
