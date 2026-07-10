import re
from dataclasses import dataclass

from fletcher.skills.skill_bank import Skill, SkillBank

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]+")


@dataclass(frozen=True)
class RetrievedSkill:
    skill: Skill
    score: float


def tokenize(text: str) -> set[str]:
    return {match.group(0).lower() for match in TOKEN_RE.finditer(text)}


class SkillRetriever:
    def __init__(self, skill_bank: SkillBank, top_k: int = 3):
        self.skill_bank = skill_bank
        self.top_k = top_k

    def retrieve(self, query: str, role: str) -> list[RetrievedSkill]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scored: list[RetrievedSkill] = []
        for skill in self.skill_bank.candidates_for(role):
            skill_tokens = tokenize(" ".join([skill.title, skill.trigger, skill.guidance, *skill.tags]))
            if not skill_tokens:
                continue
            overlap = query_tokens & skill_tokens
            union = query_tokens | skill_tokens
            score = len(overlap) / len(union)
            if score > 0:
                scored.append(RetrievedSkill(skill=skill, score=score))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[: self.top_k]


def format_skills_for_prompt(retrieved: list[RetrievedSkill]) -> str:
    if not retrieved:
        return ""

    lines = ["Reusable critic skills from prior labeled debates:"]
    for item in retrieved:
        skill = item.skill
        lines.append(
            f"- [{skill.scope}/{skill.role}] {skill.title}: "
            f"When you see {skill.trigger}, {skill.guidance}"
        )
    return "\n".join(lines) + "\n\n"
