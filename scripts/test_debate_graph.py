from fletcher.architectures.debate import build_debate_graph
from fletcher.llm.fake_client import FakeLLMClient

client = FakeLLMClient(
    canned_response='{"flagged": false, "confidence": 0.9, "reasoning": "looks fine"}'
)
app = build_debate_graph(client, roles=["conceptual", "procedural"])

result = app.invoke({
    "explanation": "A hash table stores values by converting keys into array indices using a hash function.",
    "conceptual_verdict": {},
    "procedural_verdict": {},
    "completeness_verdict": {},
    "issue_found": False,
    "final_result": "",
    "active_roles": ["conceptual", "procedural"],
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_llm_calls": 0,
    "verbose": True,
})

print("final state:", result)
