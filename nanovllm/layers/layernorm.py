import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        # 可学习参数
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()   # 转FP32高精度计算方差
        var = x.pow(2).mean(dim=-1, keepdim=True)   # 沿最后一维计算各通道方差
        x.mul_(torch.rsqrt(var + self.eps)) # RMS归一化：x / sqrt(var + eps)，原地乘减少显存
        x = x.to(orig_dtype).mul_(self.weight)  # 转回原始精度，乘可学习缩放weight
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        # 残差链路融合成单步计算
        x = x.float().add_(residual.float())
        # 保存一份新的 residual 给下一层用
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    # 入口 forward 函数 根据有无残差分别调用 rms_forward 或 add_rms_forward
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
