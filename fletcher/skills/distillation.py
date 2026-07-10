import hashlib

from fletcher.skills.skill_bank import Skill


def distill_skills_from_debate_log(log: dict) -> list[Skill]:
    """Convert one labeled debate log into reusable critic skills.

    Expected minimal shape:
    {
        "id": "...",
        "explanation": "...",
        "expected_flagged": true,
        "predicted_flagged": true,
        "debate_history": [{"round": 1, "verdicts": {...}}],
    }
    """
    expected = log.get("expected_flagged")
    predicted = log.get("predicted_flagged")
    if expected is None:
        expected = (log.get("label") or {}).get("expected_flagged")
    if predicted is None:
        predicted = (log.get("prediction") or {}).get("predicted_flagged")
    if expected is None or predicted is None:
        return []

    correct = expected == predicted
    outcome = "success" if correct else "failure"
    skills: list[Skill] = []
    final_verdicts = _final_verdicts(log)

    for critic_key, verdict in final_verdicts.items():
        role = verdict.get("role") or critic_key.split("_", 1)[0]
        flagged = bool(verdict.get("flagged", False))
        reasoning = str(verdict.get("reasoning", "")).strip()
        if not reasoning:
            continue

        if correct and expected and flagged:
            guidance = f"use this check to challenge the explanation: {reasoning}"
            title = "Catch this misconception pattern"
        elif correct and not expected and not flagged:
            guidance = f"avoid over-flagging when the explanation matches this pattern: {reasoning}"
            title = "Avoid this false-positive pattern"
        elif not correct and expected and not flagged:
            guidance = f"be stricter next time; this missed issue needed scrutiny: {reasoning}"
            title = "Missed hard-negative warning"
        elif not correct and not expected and flagged:
            guidance = f"be more conservative next time; this was likely over-flagged: {reasoning}"
            title = "False-positive warning"
        else:
            guidance = f"reuse this judgment pattern carefully: {reasoning}"
            title = "Debate judgment pattern"

        source = str(log.get("id") or log.get("sample_id") or log.get("log_id") or "unknown")
        skill_id = _stable_skill_id(source, critic_key, reasoning, outcome)
        skills.append(
            Skill(
                id=skill_id,
                scope="task_specific",
                role=role,
                title=title,
                trigger=_short_trigger(_explanation_from_log(log), reasoning),
                guidance=guidance,
                source=source,
                outcome=outcome,
                tags=[role, outcome, critic_key],
            )
        )

    return skills


def _final_verdicts(log: dict) -> dict:
    history = log.get("debate_history") or log.get("rounds") or []
    if not history:
        return {}
    return history[-1].get("verdicts", {})


def _explanation_from_log(log: dict) -> str:
    if "explanation" in log:
        return log["explanation"]
    return (log.get("input") or {}).get("explanation", "")


def _short_trigger(explanation: str, reasoning: str, max_words: int = 24) -> str:
    text = explanation.strip() or reasoning.strip()
    words = text.split()
    return " ".join(words[:max_words])


def _stable_skill_id(source: str, critic_key: str, reasoning: str, outcome: str) -> str:
    digest = hashlib.sha1(f"{source}|{critic_key}|{reasoning}|{outcome}".encode()).hexdigest()[:12]
    return f"skill_{digest}"
