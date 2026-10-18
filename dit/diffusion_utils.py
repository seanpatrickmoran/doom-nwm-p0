from __future__ import annotations
from typing import Callable, Optional
import math
import torch



def _extract(a: torch.Tensor, t: torch.Tensor, x_shape) -> torch.Tensor:
    out = a.gather(0, t.long())
    return out.view(-1, *([1] * (len(x_shape) - 1)))


class GaussianDiffusion2D:
    def __init__(self, num_steps: int = 1000, beta_schedule: str = "cosine"):
        self.num_steps = int(num_steps)
        self.beta_schedule = str(beta_schedule)
        betas = self._make_beta_schedule(self.num_steps, self.beta_schedule)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=betas.dtype), alphas_cumprod[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0)

        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_variance = posterior_var.clamp_min(1e-20)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance)
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

    def _make_beta_schedule(self, num_steps: int, name: str) -> torch.Tensor:
        if name == "linear":
            scale = 1000.0 / num_steps
            beta_start = scale * 1e-4
            beta_end = scale * 2e-2
            return torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)

        if name == "cosine":
            s = 0.008
            x = torch.linspace(0, num_steps, num_steps + 1, dtype=torch.float64)
            alphas_bar = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_bar = alphas_bar / alphas_bar[0]
            betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
            return betas.clamp(1e-4, 0.999).float()

        raise ValueError(f"Unsupported beta schedule: {name}")

    def to(self, device: torch.device) -> "GaussianDiffusion2D":
        for name, value in vars(self).items():
            if torch.is_tensor(value):
                setattr(self, name, value.to(device))
        return self

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_steps, (batch_size,), device=device)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        return _extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0 + _extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise

    def predict_x0_from_eps(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return _extract(self.sqrt_recip_alphas_cumprod, t, xt.shape) * xt - _extract(self.sqrt_recipm1_alphas_cumprod, t, xt.shape) * eps

    def p_mean_variance(self, model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], xt: torch.Tensor, t: torch.Tensor):
        eps = model_fn(xt, t)
        x0 = self.predict_x0_from_eps(xt, t, eps).clamp(-5.0, 5.0)
        mean = _extract(self.posterior_mean_coef1, t, xt.shape) * x0 + _extract(self.posterior_mean_coef2, t, xt.shape) * xt
        var = _extract(self.posterior_variance, t, xt.shape)
        log_var = _extract(self.posterior_log_variance_clipped, t, xt.shape)
        return mean, var, log_var, x0, eps

    @torch.no_grad()
    def p_sample(self, model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        mean, _var, log_var, _x0, _eps = self.p_mean_variance(model_fn, xt, t)
        noise = torch.randn_like(xt)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (xt.ndim - 1)))
        return mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def sample_loop(self, model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], shape, device: torch.device) -> torch.Tensor:
        xt = torch.randn(shape, device=device)
        for i in reversed(range(self.num_steps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            xt = self.p_sample(model_fn, xt, t)
        return xt
