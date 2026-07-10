from fletcher.architectures.full_debate import run_full_debate
from fletcher.llm.fake_client import FakeLLMClient


client = FakeLLMClient(
    canned_response=(
        '{"flagged": true, "confidence": 0.9, '
        '"reasoning": "The explanation misses collision handling.", '
        '"message_to_others": "This should be flagged."}'
    )
)

result = run_full_debate(
    client,
    roles=["conceptual", "procedural"],
    explanation="Hash tables never need collision handling.",
    max_rounds=1,
)

assert result["flagged"] is True
assert result["issue_found"] is True
assert set(result["role_results"]) == {"conceptual", "procedural"}
assert result["prompt_tokens"] > 0
assert result["completion_tokens"] > 0
assert result["llm_calls"] == 4
assert result["debate_log"]["architecture"] == "arch2_full"

print("full debate ok")
