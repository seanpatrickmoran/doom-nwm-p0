from __future__ import annotations
from typing import Optional
import torch.nn as nn
import torch
import math



def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    half = dim // 2
    device = timesteps.device
    dtype = torch.float32
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=device, dtype=dtype) / max(1, half))
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionSequenceConditioner(nn.Module):
    def __init__(
        self,
        action_dim: int,
        hidden_dim: int,
        out_dim: int,
        max_tokens: int = 32,
        depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.max_tokens = int(max_tokens)
        self.depth = int(depth)
        self.num_heads = int(num_heads)

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"ActionSequenceConditioner hidden_dim={self.hidden_dim} must be divisible by num_heads={self.num_heads}."
            )
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")

        self.in_proj = nn.Linear(self.action_dim, self.hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_tokens + 1, self.hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.depth, norm=nn.LayerNorm(self.hidden_dim))
        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, actions: Optional[torch.Tensor], action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if actions is None:
            raise ValueError("ActionSequenceConditioner expected an action tensor or an explicit zero tensor.")
        if actions.ndim != 3:
            raise ValueError(f"Expected actions with shape [B,T,A], got {tuple(actions.shape)}")

        B, T, A = actions.shape
        if A != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {A}")
        if T > self.max_tokens:
            raise ValueError(
                f"Action sequence length T={T} exceeds max_tokens={self.max_tokens}. "
                "Increase max_action_tokens when constructing the model."
            )

        if action_mask is None:
            action_mask = torch.ones((B, T), dtype=torch.bool, device=actions.device)
        else:
            if action_mask.shape != (B, T):
                raise ValueError(f"Expected action_mask shape {(B, T)}, got {tuple(action_mask.shape)}")
            action_mask = action_mask.to(device=actions.device, dtype=torch.bool)

        x = self.in_proj(actions)
        cls = self.cls_token.to(device=x.device, dtype=x.dtype).expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : T + 1].to(device=x.device, dtype=x.dtype)

        cls_valid = torch.ones((B, 1), dtype=torch.bool, device=x.device)
        valid_mask = torch.cat([cls_valid, action_mask], dim=1)
        key_padding_mask = ~valid_mask

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.out_proj(x[:, 0])


class CDiTConditioner(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        action_dim: int = 0,
        action_hidden: int = 256,
        action_seq_depth: int = 2,
        action_seq_heads: int = 4,
        action_seq_dropout: float = 0.0,
        max_action_tokens: int = 32,
        context_noise_levels: int = 0,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.context_noise_levels = int(context_noise_levels)

        self.t_embed = MLPEmbedder(self.hidden_size, self.hidden_size)
        self.h_embed = MLPEmbedder(self.hidden_size, self.hidden_size)

        self.use_actions = int(action_dim) > 0
        self.action_embed = (
            ActionSequenceConditioner(
                action_dim=action_dim,
                hidden_dim=action_hidden,
                out_dim=hidden_size,
                max_tokens=max_action_tokens,
                depth=action_seq_depth,
                num_heads=action_seq_heads,
                dropout=action_seq_dropout,
            )
            if self.use_actions
            else None
        )

        self.use_context_noise = self.context_noise_levels > 0
        self.context_noise_embed = nn.Embedding(self.context_noise_levels, self.hidden_size) if self.use_context_noise else None
        if self.context_noise_embed is not None:
            nn.init.normal_(self.context_noise_embed.weight, std=0.02)

        self.out = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

    def forward(
        self,
        diffusion_t: torch.Tensor,
        horizons: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        context_noise_bucket: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cond = self.t_embed(timestep_embedding(diffusion_t, self.hidden_size))
        cond = cond + self.h_embed(timestep_embedding(horizons, self.hidden_size))

        if self.use_actions:
            if actions is None:
                raise ValueError("This model was built with action_dim>0 but actions=None was passed.")
            cond = cond + self.action_embed(actions, action_mask)

        if self.use_context_noise:
            if context_noise_bucket is None:
                context_noise_bucket = torch.zeros_like(diffusion_t)
            context_noise_bucket = context_noise_bucket.long().clamp_(0, self.context_noise_levels - 1)
            cond = cond + self.context_noise_embed(context_noise_bucket)

        return self.out(cond)
