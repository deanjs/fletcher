import time
from typing import TYPE_CHECKING

from fletcher.agents.content_critic.conceptual import ConceptualCritic
from fletcher.agents.content_critic.procedural import ProceduralCritic
from fletcher.agents.orchestrator import DebateCriticSpec, DebateOrchestrator
from fletcher.llm.client import GenerationConfig, LLMClient
from fletcher.skills.retrieval import SkillRetriever, format_skills_for_prompt

if TYPE_CHECKING:
    from fletcher.rag.lecture_notes.retriever import LectureNoteRetriever


class SkillAugmentedCritic:
    def __init__(self, base_critic, role: str, skill_retriever: SkillRetriever):
        self.base_critic = base_critic
        self.role = role
        self.skill_retriever = skill_retriever
        self.last_skill_metrics = {
            "retrieval_ms": 0.0,
            "retrieved_skill_count": 0,
            "retrieved_skill_tokens": 0,
            "retrieved_skill_ids": [],
        }

    @property
    def last_response(self):
        return self.base_critic.last_response

    def evaluate(self, explanation: str, config: GenerationConfig | None = None, **kwargs):
        start = time.perf_counter()
        retrieved = self.skill_retriever.retrieve(explanation, role=self.role)
        retrieval_ms = (time.perf_counter() - start) * 1000
        skill_context = format_skills_for_prompt(retrieved)
        self.last_skill_metrics = {
            "retrieval_ms": retrieval_ms,
            "retrieved_skill_count": len(retrieved),
            "retrieved_skill_tokens": len(skill_context.split()),
            "retrieved_skill_ids": [item.skill.id for item in retrieved],
        }
        augmented_explanation = explanation
        if skill_context:
            augmented_explanation = f"{skill_context}Student explanation:\n{explanation}"
        return self.base_critic.evaluate(augmented_explanation, config=config, **kwargs)


def make_skill_augmented_critic(
    role: str,
    client: LLMClient,
    skill_retriever: SkillRetriever,
    persona: str = "neutral",
    retriever: "LectureNoteRetriever | None" = None,
) -> SkillAugmentedCritic:
    if role == "conceptual":
        base = ConceptualCritic(client, persona=persona, retriever=retriever)
    elif role == "procedural":
        base = ProceduralCritic(client, persona=persona, retriever=retriever)
    else:
        raise ValueError(f"Skill augmentation does not support role yet: {role}")
    return SkillAugmentedCritic(base, role=role, skill_retriever=skill_retriever)


def run_skill_persona_debate(
    client: LLMClient,
    persona_list: list[tuple[str, str]],
    explanation: str,
    skill_retriever: SkillRetriever,
    config: GenerationConfig | None = None,
    retriever: "LectureNoteRetriever | None" = None,
    max_rounds: int = 2,
    verbose: bool = False,
) -> dict:
    critics: list[DebateCriticSpec] = []
    for role, persona in persona_list:
        key = f"{role}_{persona}"
        critics.append(
            DebateCriticSpec(
                key=key,
                role=role,
                persona=persona,
                critic=make_skill_augmented_critic(
                    role,
                    client,
                    skill_retriever=skill_retriever,
                    persona=persona,
                    retriever=retriever,
                ),
            )
        )

    orchestrator = DebateOrchestrator(
        critics,
        max_rounds=max_rounds,
        config=config,
        verbose=verbose,
        log_prefix="skill_persona_debate",
        axis_config={
            "S": True,
            "N": [{"role": role, "persona": persona} for role, persona in persona_list],
        },
    )
    return orchestrator.run(explanation)
