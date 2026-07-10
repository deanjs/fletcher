import json
from pathlib import Path
from tempfile import TemporaryDirectory

from eval.metrics import evaluate_dataset
from fletcher.skills import distill_skills_from_debate_log
from fletcher.skills.skill_bank import SkillBank
from scripts.distill_debate_logs import distill_logs


dataset = [
    {
        "id": "hn_logging_001",
        "topic": "hash tables",
        "explanation": "Hash tables never need collision handling.",
        "label": True,
    }
]


def critic_fn(explanation: str):
    debate_log = {
        "architecture": "arch2",
        "axis_config": {"N": [{"role": "conceptual", "persona": "strict"}], "K": 0},
        "input": {"explanation": explanation},
        "rounds": [
            {
                "round": 1,
                "disagreement": False,
                "verdicts": {
                    "conceptual_strict": {
                        "role": "conceptual",
                        "flagged": True,
                        "confidence": 0.9,
                        "reasoning": "Hash tables still need collision handling.",
                        "message_to_others": "This should be flagged.",
                    }
                },
            }
        ],
        "final": {"flagged": True, "decision_rule": "majority_vote"},
        "metrics": {},
    }
    return True, 12.0, 100, 25, 1, debate_log


with TemporaryDirectory() as tmpdir:
    dataset_path = Path(tmpdir) / "dataset.json"
    log_path = Path(tmpdir) / "debates.jsonl"
    dataset_path.write_text(json.dumps(dataset))

    summary = evaluate_dataset(
        str(dataset_path),
        critic_fn,
        run_label="Logging Smoke / Hard Negative",
        debate_log_path=log_path,
    )

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    skills = distill_skills_from_debate_log(records[0])
    skill_bank_path = Path(tmpdir) / "skills.jsonl"
    distill_summary = distill_logs(log_path, skill_bank_path)
    skill_bank = SkillBank(skill_bank_path)

    assert summary.accuracy == 1.0
    assert len(records) == 1
    assert records[0]["label"]["expected_flagged"] is True
    assert records[0]["prediction"]["correct"] is True
    assert len(skills) == 1
    assert distill_summary["skills_created"] == 1
    assert len(skill_bank.skills) == 1

print("debate logging ok")
