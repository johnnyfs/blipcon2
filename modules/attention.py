import torch
from torch import nn

from .enums import *

class MiniAttention(nn.Module):
    def __init__(self, channels, norm_groups, nonlinearity: Optional[NonLinearity], residual: float=1.0,
                 embed_size = None):
        super().__init__()
        if embed_size is None:
            embed_size = channels
        self.norm = nn.GroupNorm(norm_groups, channels)
        self.act = get_module_for(nonlinearity)
        self.query = nn.Linear(channels, embed_size)
        self.key = nn.Linear(channels, embed_size)
        self.value = nn.Linear(channels, embed_size)
        self.project = nn.Linear(embed_size, embed_size)
        self.residual = residual

    def forward(self, x: torch.Tensor):
        input = x

        # Normalize
        x = self.act(self.norm(x))

        # Reshape and transpose
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).transpose(1, 2)

        # Calculate attention scores
        query: torch.Tensor = self.query(x)
        key: torch.Tensor = self.key(x)
        value: torch.Tensor = self.value(x)

        scores = torch.baddbmm(
            torch.empty(
                query.shape[0],
                query.shape[1],
                query.shape[1],
                dtype=x.dtype,
                device=x.device,
            ),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=1,
        )

        probs = torch.softmax(scores.float(), dim=-1).type(scores.dtype)
        x = torch.bmm(probs, value)
        x = self.project(x)
        x = x.transpose(1, 2).view(b, c, h, w)

        return x + self.residual * input