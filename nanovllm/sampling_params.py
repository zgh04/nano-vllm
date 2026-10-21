from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0    # 温度控制随机性
    max_tokens: int = 64    # 生成长度上限
    ignore_eos: bool = False    # 遇到eos是否停止

    def __post_init__(self):
        # 简化采样逻辑
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
