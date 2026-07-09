from fletcher.llm.client import LLMClient
from fletcher.llm.fake_client import FakeLLMClient


def create_llm_client(backend: str, **kwargs) -> LLMClient:
    if backend == "fake":
        return FakeLLMClient(**kwargs)
    elif backend == "hf":
        from fletcher.llm.hf_client import HFLocalClient
        return HFLocalClient(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")