import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class EvalResult:
    id: str
    topic: str
    expected: bool
    predicted: bool
    correct: bool
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int


@dataclass
class EvalSummary:
    total: int
    correct: int
    accuracy: float
    avg_latency_ms: float
    avg_prompt_tokens: float
    avg_completion_tokens: float
    results: list[EvalResult] = field(default_factory=list)


def evaluate_dataset(
    dataset_path: str,
    critic_fn: Callable[[str], tuple[bool, float, int, int]],
) -> EvalSummary:
    """
    Evaluate a critic function against a dataset.

    Args:
        dataset_path: path to JSON dataset file
        critic_fn: function that takes an explanation string and returns
                   (flagged: bool, latency_ms: float, prompt_tokens: int, completion_tokens: int)

    Returns:
        EvalSummary with accuracy, latency, and token usage stats
    """
    with open(dataset_path) as f:
        dataset = json.load(f)

    results = []

    for item in dataset:
        flagged, latency_ms, prompt_tokens, completion_tokens = critic_fn(item["explanation"])

        results.append(EvalResult(
            id=item["id"],
            topic=item["topic"],
            expected=item["label"],
            predicted=flagged,
            correct=(flagged == item["label"]),
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ))

    total = len(results)
    correct = sum(r.correct for r in results)

    return EvalSummary(
        total=total,
        correct=correct,
        accuracy=correct / total if total > 0 else 0.0,
        avg_latency_ms=sum(r.latency_ms for r in results) / total,
        avg_prompt_tokens=sum(r.prompt_tokens for r in results) / total,
        avg_completion_tokens=sum(r.completion_tokens for r in results) / total,
        results=results,
    )


def print_summary(summary: EvalSummary, label: str = "") -> None:
    header = f"--- {label} ---" if label else "--- Eval Summary ---"
    print(header)
    print(f"accuracy:              {summary.accuracy:.2%} ({summary.correct}/{summary.total})")
    print(f"avg latency:           {summary.avg_latency_ms:.1f} ms")
    print(f"avg prompt tokens:     {summary.avg_prompt_tokens:.1f}")
    print(f"avg completion tokens: {summary.avg_completion_tokens:.1f}")