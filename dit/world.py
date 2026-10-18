from __future__ import annotations

from typing import Optional
import torch
from torch import nn
from .diffusion_utils import GaussianDiffusion2D


def rounded_ddim_timesteps(train_steps: int, sample_steps: int) -> list[int]:
    sample_steps = max(1, min(int(sample_steps), int(train_steps)))
    if sample_steps == train_steps:
        return list(range(train_steps))
    vals = torch.linspace(0, train_steps - 1, sample_steps).round().long().tolist()
    return list(dict.fromkeys(int(t) for t in vals))


@torch.no_grad()
def rounded_ddim_sample_loop(diffusion, model_fn, shape, device, *, num_sample_steps: int,
                             eta: float = 0.0, clip_x0=(-5.0, 5.0), init_noise=None):
    xt = torch.randn(shape, device=device) if init_noise is None else init_noise.to(device=device, dtype=torch.float32)
    if tuple(xt.shape) != tuple(shape):
        raise ValueError(f"initial noise: expected {shape}, got {tuple(xt.shape)}")
    schedule = rounded_ddim_timesteps(diffusion.num_steps, num_sample_steps)
    for idx in range(len(schedule) - 1, -1, -1):
        ti = schedule[idx]
        prev = schedule[idx - 1] if idx > 0 else -1
        t = torch.full((shape[0],), ti, device=device, dtype=torch.long)
        eps = model_fn(xt, t)
        x0 = diffusion.predict_x0_from_eps(xt, t, eps)
        if clip_x0 is not None:
            x0 = x0.clamp(float(clip_x0[0]), float(clip_x0[1]))
        at = diffusion.alphas_cumprod[ti].to(device=device, dtype=xt.dtype)
        ap = diffusion.alphas_cumprod[prev].to(device=device, dtype=xt.dtype) if prev >= 0 else torch.tensor(1.0, device=device, dtype=xt.dtype)
        sigma = float(eta) * torch.sqrt(((1-ap)/(1-at)).clamp_min(0)) * torch.sqrt((1-at/ap).clamp_min(0))
        direction = torch.sqrt((1-ap-sigma.square()).clamp_min(0)) * eps
        noise = torch.randn_like(xt) if prev >= 0 and eta > 0 else torch.zeros_like(xt)
        xt = torch.sqrt(ap) * x0 + direction + sigma * noise
    return xt


class WorldModel(nn.Module):
    """One frozen VAE/DiT; used for steered lookahead and visible generation."""
    def __init__(self, vae, dit, diffusion: GaussianDiffusion2D, *, device,
                 vae_config, dit_config, diffusion_config, resolve):
        super().__init__()
        self.vae, self.dit = vae, dit
        self.diffusion = diffusion
        self.device = torch.device(device)
        self.vae_config, self.dit_config, self.diffusion_config = vae_config, dit_config, diffusion_config
        self.resolve = resolve

    @torch.no_grad()
    def encode_obs(self, obs_btchw: torch.Tensor, *, sample_latent=False) -> torch.Tensor:
        obs = obs_btchw.to(self.device)
        if obs.dtype == torch.uint8:
            obs = obs.float() / 255.0
        else:
            obs = obs.float()
            if obs.numel() and float(obs.detach().amax().item()) > 2:
                obs = obs / 255.0
        x = obs.permute(0, 2, 1, 3, 4).contiguous() * 2.0 - 1.0
        mu, logvar = self.vae.encode(x, scale=(0.0, 1.0))
        z = self.vae.reparameterize(mu, logvar) if sample_latent else mu
        return z.permute(0, 2, 1, 3, 4).contiguous()

    @torch.no_grad()
    def decode_latent(self, z_bzhw, *, output_size):
        x = self.vae.decode(z_bzhw.to(self.device).unsqueeze(2), scale=(0.0, 1.0), output_size=output_size)
        return (x.squeeze(2) * 0.5 + 0.5).clamp(0.0, 1.0)

    @torch.no_grad()
    def predict_latent(self, context_latents, actions, *, horizon=1, action_mask=None,
                       context_noise_bucket=None, sampler="ddim", ddim_steps=None, eta=0.0,
                       init_noise=None):
        ctx = context_latents.to(self.device, dtype=torch.float32)
        actions = actions.to(self.device, dtype=torch.float32)
        mask = torch.ones(actions.shape[:2], device=self.device, dtype=torch.bool) if action_mask is None else action_mask.to(self.device).bool()
        b, _tc, z, h, w = ctx.shape
        horizons = torch.full((b,), int(horizon), device=self.device, dtype=torch.long)
        bucket = (torch.full((b,), int(self.diffusion_config.get("context_noise_bucket", 0)), device=self.device, dtype=torch.long)
                  if context_noise_bucket is None else context_noise_bucket.to(self.device, dtype=torch.long))
        def model_fn(xt, t):
            return self.dit(noisy_target=xt, context_latents=ctx, diffusion_t=t,
                            horizons=horizons, actions=actions, action_mask=mask,
                            context_noise_bucket=bucket)
        shape = (b, z, h, w)
        if sampler == "ddpm":
            if init_noise is not None:
                raise ValueError("init_noise is supported only for DDIM")
            return self.diffusion.sample_loop(model_fn=model_fn, shape=shape, device=self.device)
        if sampler != "ddim":
            raise ValueError(f"unknown sampler {sampler}")
        return rounded_ddim_sample_loop(self.diffusion, model_fn, shape, self.device,
                                        num_sample_steps=int(ddim_steps or self.diffusion_config.get("ddim_steps", 50)),
                                        eta=float(eta), clip_x0=(-5.0, 5.0), init_noise=init_noise)
