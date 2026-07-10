from fletcher.finetuning import build_training_records, compute_skill_reward
from fletcher.serving import ServingConfig, estimate_round_latency_ms


log = {
    "log_id": "sample_001",
    "input": {"explanation": "Hash tables never need collision handling."},
    "prediction": {"correct": True},
    "rounds": [
        {
            "verdicts": {
                "conceptual_strict": {
                    "role": "conceptual",
                    "flagged": True,
                    "reasoning": "Collision handling is required.",
                    "confidence": 0.9,
                    "skill_metrics": {"retrieved_skill_ids": ["skill_001"]},
                }
            }
        }
    ],
}

reward = compute_skill_reward(log)
records = build_training_records([log])
latency = estimate_round_latency_ms(
    100.0,
    critic_count=5,
    config=ServingConfig(execution_mode="asyncio", max_concurrent_critics=2),
)

assert reward > 1.0
assert records[0]["skill_ids"] == ["skill_001"]
assert latency == 300.0

print("policy and serving ok")
