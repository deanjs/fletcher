from dataclasses import dataclass
from typing import Literal


ExecutionMode = Literal["sequential", "asyncio", "vllm"]


@dataclass(frozen=True)
class ServingConfig:
    execution_mode: ExecutionMode = "sequential"
    max_concurrent_critics: int = 1
    use_kv_cache: bool = True
    use_paged_attention: bool = False
    use_flash_attention: bool = False


def estimate_round_latency_ms(per_call_latency_ms: float, critic_count: int, config: ServingConfig) -> float:
    if critic_count <= 0:
        return 0.0
    if config.execution_mode == "sequential":
        return per_call_latency_ms * critic_count
    parallel_width = max(1, min(config.max_concurrent_critics, critic_count))
    batches = (critic_count + parallel_width - 1) // parallel_width
    return per_call_latency_ms * batches
