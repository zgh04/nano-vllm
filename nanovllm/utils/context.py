from dataclasses import dataclass
import torch

# 上下文管理
# 生产者：ModelRunner.prepare_prefill()，ModelRunner.prepare_decode()
# 消费者：Attention（slot_mapping, block_tables, cu_seqlens），ParallelLMHead（prefill时取最后位置），ModelRunner.run_model（CUDA graph 回放时填入 slot_mapping）
@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    # 累计序列长度 构成的张量，作用？
    # 示例：batch 内两条序列长度 3、5 → [0, 3, 8]
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    # 当前批次里单条请求最大序列长度
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    # PagedAttention 核心张量 记录每个 token 到 KVcache 块内的 slot 位置
    slot_mapping: torch.Tensor | None = None
    # batch 里每条请求 输入 + 生成 的总上下文长度 decode 用
    context_lens: torch.Tensor | None = None
    # KVcache 块表 记录每个请求占用哪些 物理缓存块 padding 到等长
    block_tables: torch.Tensor | None = None

# 模块顶层创建唯一全局 Context 对象
_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    # 声明要修改顶层全局变量
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

# 重置上下文
def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
