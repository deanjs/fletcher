from fletcher.llm.factory import create_llm_client
from fletcher.agents.content_critic.conceptual import ConceptualCritic
from fletcher.agents.content_critic.procedural import ProceduralCritic

fake_json_response = '{"flagged": true, "confidence": 0.8, "reasoning": "Missing collision handling."}'

client = create_llm_client("fake", canned_response=fake_json_response)

conceptual = ConceptualCritic(client)
procedural = ProceduralCritic(client)

explanation = "A hash table stores values by converting keys into array indices using a hash function."

conceptual_verdict = conceptual.evaluate(explanation)
procedural_verdict = procedural.evaluate(explanation)

print("conceptual verdict:", conceptual_verdict)
print("procedural verdict:", procedural_verdict)