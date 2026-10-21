from functools import lru_cache
import torch
from torch import nn


# 核心旋转公式：
# 数学意义：把向量 (x1,x2) 在二维平面旋转 θ
def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    # RoPE 缓存预计算层，使得模型推理时只查表不计算
    def __init__(
        self,
        head_size: int, # 单注意力头维度
        rotary_dim: int,    # 使用 RoPE 的维度
        max_position_embeddings: int,   # 模型最大上下文长度
        base: float,    # RoPE 基底常数
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size  # 所有维度都使用 RoPE
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))   # 逆频率
        t = torch.arange(max_position_embeddings, dtype=torch.float)    # token 绝对位置：0,1,2,...,max_pos-1
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        # 提前计算全部位置，全部维度的 sin 和 cos 推理时只查表不计算
        cos = freqs.cos()
        sin = freqs.sin()
        # cat：拼接 sin 和 cos 一次索引全部取出
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        # persistent=False：保存模型.bin权重时不存储这个超大缓存，加载时自动重新生成
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1) # 从缓存中取出并拆分 sin 和 cos
        # 为什么 Q K 分别调用 apply_rotary_emb 不合并？
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key

# 全局获取入口
# 项目内所有attention.py层统一调用get_rope(xxx)拿到单例
@lru_cache(1)   # to 最多缓存一份实例
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
