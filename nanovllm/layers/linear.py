import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

# 自定义多种 linear 层 实现张量并行（TP）

# 保证模型维度可以被 TP 卡数整除
def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim    # 切分维度 0 列并行 1 行并行 None 不切分
        self.tp_rank = dist.get_rank()  # 当前卡 TP 编号 
        self.tp_size = dist.get_world_size()    # TP 总卡数
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        # 自定义权重加载钩子，不同类型线性层使用不同 weight_loader
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

# 直接复制型线性层，每张 GPU 卡都保存完整权重矩阵
# TP 切分收益低 通信开销高时使用
class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    # 直接完整拷贝权重
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 标准原生 linear 无分布式通信操作
        return F.linear(x, self.weight, self.bias)

# 基础列并行层，用来做 QKV / GateUP
# Y = X * W ，W = （W1，W2），Y = （X*W1，X*W2）
class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # 输入维度不变，输出维度被 TP 均分
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        # 从完整权重截取当前 rank 对应的部分权重
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    # 注意：各卡之间独立计算，无通信。上层 Attention/FFN 负责拼接结果
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

# 多分支合并列并行？
# LLaMA FFN Gate（分支0） + Up（分支1） 两个线性层合并为一个大矩阵
# Y = X * （Wg，Wu），（Wg，Wu）= （Wg0，Wg1，Wu0，Wu1），GPU0 = （Wg0，Wu0），GPU1 = （Wg1，Wu1），Y = concat（X*GPU0，X*GPU1）
class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        # 总输出 = 多个分支输出之和
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


# 适配标准 MHA / GQA 模型 QKV 合并权重 自动处理Q/KV 按头 TP 切分
# Q 头总数均分至各 TP 卡，KV 头总数均分至各 TP 卡。QKV 不在一起是因为 GQA 中 KV 头少于 Q 头
# Y = X * （Wq，Wk，Wv），（Wq，Wk,Wv）= （Wq0，Wq1，Wk0，Wk1，Wv0，Wv1），GPU0 = （Wq0，Wk0,Wv0），GPU1 = （Wq1，Wk1,Wv1），Y = concat（X*GPU0，X*GPU1）
class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        # 当前卡本地头数 = 总头数 / tp_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        # 总输出维度 = Q + K + V
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 全局权重分片后取当前 rank
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


# 行并行线性层
# Y = X * W，X = （X1，X2），W = （W1，W2）.T，Y = X1 * W1 + X2 * W2
# 适用于 Attention 输出 O Proj、FFN Down 下投影
class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # 输入维度被 TP 均分，输出维度不变
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        # bias 一维不切分，直接拷贝
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 只有 rank0 持有 bias 其他卡禁用 bias 避免重复
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            # 原地全局求和 同步所有 GPU 结果
            dist.all_reduce(y)
        return y
