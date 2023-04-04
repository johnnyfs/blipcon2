from typing import Optional
import torch


class MiniDist:
    def __init__(self, params):
        self.params = params
        # Split the latent channels into mean and log variance
        mean, logvar = torch.chunk(params, 2, dim=1)
        self.mean = mean
        # Prevent numerical instability
        self.logvar = torch.clamp(logvar, min=-10, max=10)
        self.std = torch.exp(0.5 * logvar)

    def sample(self, generator: Optional[torch.Generator]=None):
        sample = torch.randn(self.mean.shape, device=self.params.device, generator=generator)
        return self.mean + self.std * sample
