from fletcher.agents.schemas import CriticVerdict

verdict = CriticVerdict(
    role="conceptual",
    flagged=True,
    reasoning="The explanation omits collision handling, a core part of hash table implementation.",
    confidence=0.85,
)

print(verdict)
print(verdict.model_dump_json(indent=2))