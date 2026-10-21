import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


# Triton 存储 KV 缓存内核，PagedAttention 核心写入算子
@triton.jit
def store_kvcache_kernel(
    key_ptr,    # 当前 step 新计算出的 K 张量起始指针
    key_stride, # 一个 token 占多少元素
    value_ptr,  # 当前 step 新计算出的 V 张量起始指针
    value_stride,
    k_cache_ptr,    # 全局分页 KV 缓存池指针
    v_cache_ptr,
    slot_mapping_ptr,   # 映射张量指针，shape[N]，每个元素代表当前要写入的 缓存 slot 编号
    D: tl.constexpr,    # 单个 slot 存储的 KV 总维度
):
    # 1.GPU 并行地将当前 step 新生成的 KV张量写入 分页式物理 KV 缓存
    # 启动 N 个并行进程 每个 GPU 线程块对应一个 token
    idx = tl.program_id(0)
    # 读取当前 token 对应的 物理缓存 slot 编号
    slot = tl.load(slot_mapping_ptr + idx)
    # slot 是 warmup 阶段的占位符
    if slot == -1: return
    # 计算当前 token KV 在输入张量中的显存偏移
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    # 2.通过 slot_mapping 将 逻辑 token 位置 映射到 显存中碎片化的 KV 缓存槽位
    # 写入分页缓存池
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


# ...
def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    # N 当前批次总 token 数量 D 单 slot 存储维度=kv头数 * 单头维度，传给 triton 作为 constexpr
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    # 这些断言用来确保内存连续
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    # 启动 N 个并行线程，每个处理一个 token
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,  # Q 头总数
        head_dim,   # 单头特征维度
        scale,  # 注意力 softmax 缩放因子
        num_kv_heads,   # KV 头数量
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # ...
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context() # 读取上下文
        k_cache, v_cache = self.k_cache, self.v_cache
        # 将当前 KV 写入分页 KVcache 缓存池
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache 复用
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
