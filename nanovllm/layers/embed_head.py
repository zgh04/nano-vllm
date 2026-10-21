import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context

# 并行词嵌入
class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,    # 全局总词表大小
        embedding_dim: int, # 隐藏层维度
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        # 词表能被 TP 卡均分
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        # 单卡本地存储的词数量
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        # 当前卡负责的全局词表区间（用 id 表示）
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        # 自己的分片词向量
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        # 自定义权重分片加载钩子
        self.weight.weight_loader = self.weight_loader

    # 权重分片加载
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    # 输入x：一批全局 token id
    # 示例：x = [5,17000,12000],tp = 2,V = 32000
    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            # 标记哪些 token_id 属于自己负责的区间
            # rank0:mask = [True,False,True],
            # rank1:mask = [False,True,False]
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 全局 id 转相对于自身起始的相对位置 id
            # rank0:x = [5,0,12000],
            # rank1:x = [0,1000,0]
            x = mask * (x - self.vocab_start_idx)
        # 取出当前卡负责的 token 向量 无关 token 输出全 0 向量
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            # 通信机制，所有卡原地求和得到完整 词嵌入向量
            dist.all_reduce(y)
        return y


# 并行输出头，用于计算logits = X * W.T,其中 W：[V,H]
class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        # 不使用 bias
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    # 分 Prefill 和 Decode
    def forward(self, x: torch.Tensor):
        context = get_context()
        # prefill 阶段 只取每个序列最后一个 token 作预测
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            # 收集所有卡分片的 logits，仅 rank0 拼接完整全局词表 logits
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            # 每张卡输出独立一段词表 logits 不能像 VocabParallelEmbedding 一样求和
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
