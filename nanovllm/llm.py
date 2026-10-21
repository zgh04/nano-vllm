from nanovllm.engine.llm_engine import LLMEngine

# LLM类 等价于 LLMEngine 目的是保持与 VLLM 一样的 API
class LLM(LLMEngine):
    pass
