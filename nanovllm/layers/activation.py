import torch
from torch import nn
import torch.nn.functional as F

# ??
# 门控激活：gate × up 的融合投影 + chunk 分割，一个矩阵乘法完成两个投影
# kernel 融合：@torch.compile 把 Python 描述的计算图编译成单个 GPU kernel，减少显存带宽浪费
class SiluAndMul(nn.Module):

    # 编译成单个 GPU kernel 减少显存带宽浪费
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
