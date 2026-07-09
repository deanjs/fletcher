from fletcher.llm.factory import create_llm_client
from fletcher.llm.message import Message
from fletcher.llm.client import GenerationConfig

client = create_llm_client("fake", canned_response="This is a test response.")

messages = [
    Message(role="system", content="You are a helpful assistant."),
    Message(role="user", content="What is 1+1?"),
]

response = client.generate(messages, config=GenerationConfig(max_new_tokens=100))

print("text:", response.text)
print("prompt_tokens:", response.prompt_tokens)
print("completion_tokens:", response.completion_tokens)
print("latency_ms:", response.latency_ms)