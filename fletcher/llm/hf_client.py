import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from fletcher.llm.client import GenerationConfig, LLMClient, LLMResponse
from fletcher.llm.message import Message


class HFLocalClient(LLMClient):
    def __init__(
        self,
        model_name: str = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        device_map: str = "auto",
    ):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map=device_map,
        )

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
            do_sample=True,
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