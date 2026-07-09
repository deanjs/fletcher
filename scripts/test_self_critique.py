from fletcher.llm.factory import create_llm_client
from fletcher.architectures.self_critique import SelfCritique

client = create_llm_client("fake", canned_response="The explanation is mostly correct, but misses the collision handling mechanism.")
critic = SelfCritique(client)

explanation = "A hash table stores values by converting keys into array indices using a hash function."

result = critic.critique(explanation)

print("critique result:", result)
