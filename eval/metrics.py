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
    llm_calls: int


@dataclass
class EvalSummary:
    total: int
    correct: int
    accuracy: float
    avg_latency_ms: float
    avg_prompt_tokens: float
    avg_completion_tokens: float
    avg_llm_calls: float
    results: list[EvalResult] = field(default_factory=list)


def evaluate_dataset(
    dataset_path: str,
    critic_fn: Callable[[str], tuple],
    run_label: str = "",
    verbose: bool = False,
    limit: int | None = None,
    debate_log_path: str | Path | None = None,
) -> EvalSummary:
    with open(dataset_path) as f:
        dataset = json.load(f)

    if limit is not None:
        dataset = dataset[:limit]

    results = []

    for index, item in enumerate(dataset, start=1):
        if verbose:
            label_text = f"[{run_label}] " if run_label else ""
            print(
                f"{label_text}[eval] Sample {index}/{len(dataset)}: "
                f"{item['id']} ({item['topic']})",
                flush=True,
            )
        critic_result = critic_fn(item["explanation"])
        flagged, latency_ms, prompt_tokens, completion_tokens, llm_calls = critic_result[:5]
        debate_log = critic_result[5] if len(critic_result) > 5 else None

        results.append(EvalResult(
            id=item["id"],
            topic=item["topic"],
            expected=item["label"],
            predicted=flagged,
            correct=(flagged == item["label"]),
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            llm_calls=llm_calls,
        ))
        if debate_log_path is not None and debate_log is not None:
            _append_debate_log(
                debate_log_path,
                debate_log,
                item=item,
                run_label=run_label,
                predicted=flagged,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                llm_calls=llm_calls,
            )

        if verbose:
            print(flush=True)

    total = len(results)
    correct = sum(r.correct for r in results)

    return EvalSummary(
        total=total,
        correct=correct,
        accuracy=correct / total if total > 0 else 0.0,
        avg_latency_ms=sum(r.latency_ms for r in results) / total,
        avg_prompt_tokens=sum(r.prompt_tokens for r in results) / total,
        avg_completion_tokens=sum(r.completion_tokens for r in results) / total,
        avg_llm_calls=sum(r.llm_calls for r in results) / total,
        results=results,
    )


def print_summary(summary: EvalSummary, label: str = "") -> None:
    header = f"--- {label} ---" if label else "--- Eval Summary ---"
    print(header, flush=True)
    print(f"accuracy:              {summary.accuracy:.2%} ({summary.correct}/{summary.total})", flush=True)
    print(f"avg latency:           {summary.avg_latency_ms:.1f} ms", flush=True)
    print(f"avg prompt tokens:     {summary.avg_prompt_tokens:.1f}", flush=True)
    print(f"avg completion tokens: {summary.avg_completion_tokens:.1f}", flush=True)
    print(f"avg llm calls:         {summary.avg_llm_calls:.1f}", flush=True)
    print(flush=True)


def _append_debate_log(
    path: str | Path,
    debate_log: dict,
    item: dict,
    run_label: str,
    predicted: bool,
    latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    llm_calls: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = dict(debate_log)
    record["log_id"] = f"{item['id']}__{_slug(run_label)}"
    record["sample_id"] = item["id"]
    record["run_label"] = run_label
    record["input"] = {
        **record.get("input", {}),
        "topic": item.get("topic", ""),
        "explanation": item["explanation"],
    }
    record["label"] = {
        "expected_flagged": item["label"],
        "source": "hard_negative" if item["label"] else "normal",
    }
    record["prediction"] = {
        "predicted_flagged": predicted,
        "correct": predicted == item["label"],
    }
    record["metrics"] = {
        **record.get("metrics", {}),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "llm_calls": llm_calls,
        "latency_ms": latency_ms,
    }
    if "final" in record:
        record["final"] = {
            **record["final"],
            "correct": predicted == item["label"],
        }

    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _slug(text: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    return "_".join(part for part in slug.split("_") if part)
