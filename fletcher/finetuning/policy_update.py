from dataclasses import dataclass


@dataclass(frozen=True)
class RewardWeights:
    outcome: float = 1.0
    skill_reuse: float = 0.2
    schema_valid: float = 0.1
    anomaly_penalty: float = -0.5


def compute_skill_reward(log: dict, weights: RewardWeights | None = None) -> float:
    """Compute the scalar reward used to prepare future SFT/GRPO data.

    This does not run optimization. It turns an evaluated debate trajectory
    into a stable reward signal that a trainer can consume later.
    """

    weights = weights or RewardWeights()
    prediction = log.get("prediction", {})
    final = log.get("final", {})
    correct = bool(prediction.get("correct", final.get("correct", False)))
    schema_valid = _schema_valid(log)
    used_skill = _used_skill(log)

    reward = weights.outcome if correct else -weights.outcome
    if used_skill and correct:
        reward += weights.skill_reuse
    if schema_valid:
        reward += weights.schema_valid
    else:
        reward += weights.anomaly_penalty
    return reward


def build_training_records(logs: list[dict], weights: RewardWeights | None = None) -> list[dict]:
    records = []
    for log in logs:
        records.append({
            "log_id": log.get("log_id") or log.get("id") or log.get("sample_id", "unknown"),
            "input": log.get("input", {}),
            "label": log.get("label", {}),
            "prediction": log.get("prediction", {}),
            "reward": compute_skill_reward(log, weights=weights),
            "skill_ids": _skill_ids(log),
        })
    return records


def _schema_valid(log: dict) -> bool:
    rounds = log.get("rounds", [])
    if not rounds and "roles" in log:
        rounds = [
            role_round
            for role_log in log["roles"].values()
            for role_round in role_log.get("rounds", [])
        ]
    if not rounds:
        return False
    for round_record in rounds:
        for verdict in round_record.get("verdicts", {}).values():
            if not {"role", "flagged", "reasoning", "confidence"} <= set(verdict):
                return False
    return True


def _used_skill(log: dict) -> bool:
    return bool(_skill_ids(log))


def _skill_ids(log: dict) -> list[str]:
    ids: list[str] = []
    rounds = log.get("rounds", [])
    if not rounds and "roles" in log:
        rounds = [
            role_round
            for role_log in log["roles"].values()
            for role_round in role_log.get("rounds", [])
        ]
    for round_record in rounds:
        for verdict in round_record.get("verdicts", {}).values():
            metrics = verdict.get("skill_metrics", {})
            ids.extend(metrics.get("retrieved_skill_ids", []))
    return ids
