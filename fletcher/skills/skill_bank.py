import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


SkillScope = Literal["general", "task_specific"]


@dataclass(frozen=True)
class Skill:
    id: str
    scope: SkillScope
    role: str
    title: str
    trigger: str
    guidance: str
    source: str
    outcome: Literal["success", "failure"]
    tags: list[str] = field(default_factory=list)

    def to_record(self) -> dict:
        return {
            "id": self.id,
            "scope": self.scope,
            "role": self.role,
            "title": self.title,
            "trigger": self.trigger,
            "guidance": self.guidance,
            "source": self.source,
            "outcome": self.outcome,
            "tags": self.tags,
        }

    @classmethod
    def from_record(cls, record: dict) -> "Skill":
        return cls(
            id=record["id"],
            scope=record["scope"],
            role=record["role"],
            title=record["title"],
            trigger=record["trigger"],
            guidance=record["guidance"],
            source=record["source"],
            outcome=record["outcome"],
            tags=list(record.get("tags", [])),
        )


class SkillBank:
    """Small persistent skill library.

    The hierarchy is represented by `scope`: general skills are available to
    every critic, while task-specific skills are filtered by critic role.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._skills: dict[str, Skill] = {}
        self.load()

    @property
    def skills(self) -> list[Skill]:
        return list(self._skills.values())

    def load(self) -> None:
        self._skills = {}
        if not self.path.exists():
            return

        with self.path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                skill = Skill.from_record(json.loads(line))
                self._skills[skill.id] = skill

    def add(self, skill: Skill) -> None:
        self._skills[skill.id] = skill

    def extend(self, skills: list[Skill]) -> None:
        for skill in skills:
            self.add(skill)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w") as f:
            for skill in self._skills.values():
                f.write(json.dumps(skill.to_record(), sort_keys=True) + "\n")

    def candidates_for(self, role: str) -> list[Skill]:
        return [
            skill
            for skill in self._skills.values()
            if skill.scope == "general" or skill.role == role
        ]
