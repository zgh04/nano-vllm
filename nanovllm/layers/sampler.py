import torch
from torch import nn

# 采样方法:
# 使用 Gumbel-max 而不是 multinomial
# clamp_min_ 防止除 0
# 禁止 temperature 为 0 为了简化
class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
