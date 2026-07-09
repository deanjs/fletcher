from fletcher.llm.client import GenerationConfig, LLMClient, LLMResponse
from fletcher.llm.message import Message


class SelfCritique:
    def __init__(self, client: LLMClient):
        self.client = client
        self.last_response: LLMResponse | None = None

    def critique(self, explanation: str, config: GenerationConfig | None = None) -> str:
        config = config or GenerationConfig()

        initial_messages = [
            Message(
                role="system",
                content=(
                    "You are a content fidelity critic for a computer science course. "
                    "Evaluate the student's explanation below for factual, conceptual, and procedural "
                    "accuracy. Point out any misconceptions or missing key ideas. "
                    "Do not evaluate English proficiency or writing style."
                ),
            ),
            Message(
                role="user",
                content=f"Student explanation:\n\n{explanation}",
            ),
        ]
        initial_response = self.client.generate(initial_messages, config=config)

        review_messages = [
            Message(
                role="system",
                content=(
                    "You are reviewing your own critique of a student's technical explanation. "
                    "Check whether your critique was strict enough, whether you missed any factual, "
                    "conceptual, or procedural issue, and whether any claim needs correction. "
                    "Be explicit about weaknesses in the critique."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"Student explanation:\n\n{explanation}\n\n"
                    f"Your initial critique:\n\n{initial_response.text}\n\n"
                    "Did you miss anything important or judge too leniently?"
                ),
            ),
        ]
        review_response = self.client.generate(review_messages, config=config)

        revision_messages = [
            Message(
                role="system",
                content=(
                    "You are revising a critique of a student's technical explanation. "
                    "Produce the final critique by incorporating valid self-review findings. "
                    "Keep all legitimate concerns, add any missed issues, and remove unsupported ones. "
                    "Do not mention the self-review process."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"Student explanation:\n\n{explanation}\n\n"
                    f"Initial critique:\n\n{initial_response.text}\n\n"
                    f"Self-review notes:\n\n{review_response.text}\n\n"
                    "Write the final revised critique."
                ),
            ),
        ]
        final_response = self.client.generate(revision_messages, config=config)

        self.last_response = LLMResponse(
            text=final_response.text,
            prompt_tokens=(
                initial_response.prompt_tokens
                + review_response.prompt_tokens
                + final_response.prompt_tokens
            ),
            completion_tokens=(
                initial_response.completion_tokens
                + review_response.completion_tokens
                + final_response.completion_tokens
            ),
            latency_ms=(
                initial_response.latency_ms
                + review_response.latency_ms
                + final_response.latency_ms
            ),
        )
        return final_response.text
