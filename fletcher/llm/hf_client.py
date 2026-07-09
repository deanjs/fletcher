import logging
import time

from transformers import AutoModelForCausalLM, AutoTokenizer

from fletcher.llm.client import GenerationConfig, LLMClient, LLMResponse
from fletcher.llm.message import Message

# Silence the "Both `max_new_tokens` and `max_length` seem to have been set"
# warning. We always pass max_new_tokens explicitly and intentionally clear
# generation_config.max_length below, so the warning is a false positive.
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)


class HFLocalClient(LLMClient):
    def __init__(
        self,
        model_name: str = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        device_map: str = "auto",
    ):
        # unsloth/Qwen2.5-7B-Instruct-bnb-4bit already ships its own
        # quantization_config (bnb 4-bit); passing a second one here just
        # triggers a "the model's own quantization_config will be used"
        # warning, so we let the model's bundled config apply.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
        )
        # The model's default generation_config ships with a fixed max_length
        # (e.g. 32768). Leaving it set makes `generate()` warn that
        # max_new_tokens takes precedence on every call; clear it so only
        # max_new_tokens governs output length.
        self.model.generation_config.max_length = None

    def generate(
        self,
        messages: list[Message],
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        config = config or GenerationConfig()

        chat = [{"role": m.role, "content": m.content} for m in messages]
        prompt = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        start = time.perf_counter()
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=False,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        completion_ids = output_ids[0][prompt_tokens:]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(completion_ids),
            latency_ms=latency_ms,
        )
