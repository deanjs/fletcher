from pathlib import Path
from tempfile import TemporaryDirectory

from fletcher.architectures.skill_augmented import run_skill_persona_debate
from fletcher.llm.fake_client import FakeLLMClient
from fletcher.skills import SkillBank, SkillRetriever, distill_skills_from_debate_log


debate_log = {
    "id": "hn_hash_collision_001",
    "explanation": "Hash tables never need collision handling because every key maps to a unique index.",
    "expected_flagged": True,
    "predicted_flagged": True,
    "debate_history": [
        {
            "round": 1,
            "verdicts": {
                "conceptual_strict": {
                    "role": "conceptual",
                    "flagged": True,
                    "confidence": 0.92,
                    "reasoning": "Hash functions can map different keys to the same index, so collisions must be handled.",
                    "message_to_others": "This is a conceptual misconception about hash functions.",
                }
            },
        }
    ],
}


with TemporaryDirectory() as tmpdir:
    bank = SkillBank(Path(tmpdir) / "skills.jsonl")
    skills = distill_skills_from_debate_log(debate_log)
    bank.extend(skills)
    bank.save()

    reloaded = SkillBank(Path(tmpdir) / "skills.jsonl")
    retrieved = SkillRetriever(reloaded, top_k=1).retrieve(
        "A student says hash tables do not need collision handling.",
        role="conceptual",
    )
    debate_result = run_skill_persona_debate(
        FakeLLMClient(
            canned_response=(
                '{"flagged": true, "confidence": 0.9, '
                '"reasoning": "collision handling is required", '
                '"message_to_others": "This should be flagged."}'
            )
        ),
        [("conceptual", "strict"), ("conceptual", "merciful")],
        "Hash tables do not need collision handling.",
        skill_retriever=SkillRetriever(reloaded, top_k=1),
        max_rounds=1,
    )

    assert len(skills) == 1
    assert len(retrieved) == 1
    assert retrieved[0].skill.role == "conceptual"
    assert debate_result["flagged"] is True
    assert debate_result["debate_log"]["metrics"]["retrieved_skill_count"] == 2
    assert debate_result["debate_log"]["metrics"]["retrieved_skill_tokens"] > 0
    assert (
        debate_result["debate_log"]["rounds"][0]["verdicts"]["conceptual_strict"]["skill_metrics"][
            "retrieved_skill_count"
        ]
        == 1
    )

print("skill pipeline ok")
