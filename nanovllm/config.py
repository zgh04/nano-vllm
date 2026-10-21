import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    # 使用 -1 作占位符 因为 EOS token ID 在加载 tokenizer 后才知道 
    # 而 LLMEngine.__init__ 里：Config 对象创建在前，tokenizer 加载在后
    eos: int = -1
    # flash-attention 的 block_table 对内存对齐有要求
    kvcache_block_size: int = 256
    # 初始化为 -1 真实值在 ModelRunner 启动后根据 GPU 显存反推
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        # 使用 transformers库 来读取模型配置（简化本地代码） 代价是只支持 HF 格式的模型  
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
