from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditioning import CDiTConditioner
from .patching import PatchEmbed2D, build_2d_sincos_pos_embed, unpatchify_2d



def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU(approximate="tanh")
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, dropout: float = 0.0):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        if dim % self.num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, N, C)
        return self.proj(y)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, dropout: float = 0.0):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        if dim % self.num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        M = context.shape[1]
        q = self.to_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, N, C)
        return self.proj(y)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(dim, num_heads=num_heads, dropout=dropout)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, dropout=dropout)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 9))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        s1, sc1, g1, s2, sc2, g2, s3, sc3, g3 = self.adaLN(cond).chunk(9, dim=-1)
        x = x + g1.unsqueeze(1) * self.self_attn(modulate(self.norm1(x), s1, sc1))
        x = x + g2.unsqueeze(1) * self.cross_attn(modulate(self.norm2(x), s2, sc2), context)
        x = x + g3.unsqueeze(1) * self.mlp(modulate(self.norm3(x), s3, sc3))
        return x


class ActionConditionedDiT(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        action_dim: int = 0,
        input_size: int = 16,
        patch_size: int = 2,
        hidden_size: int = 768,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_context_frames: int = 8,
        learn_context_time_embed: bool = True,
        action_hidden: int = 256,
        action_seq_depth: int = 2,
        action_seq_heads: int = 4,
        action_seq_dropout: float = 0.0,
        max_action_tokens: int = 32,
        context_noise_levels: int = 8,
    ):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.action_dim = int(action_dim)
        self.input_size = int(input_size)
        self.patch_size = int(patch_size)
        self.hidden_size = int(hidden_size)
        self.depth = int(depth)
        self.max_context_frames = int(max_context_frames)
        self.context_noise_levels = int(context_noise_levels)

        self.patch_embed = PatchEmbed2D(self.latent_channels, self.hidden_size, patch_size=self.patch_size)
        grid = self.input_size // self.patch_size
        pos = build_2d_sincos_pos_embed(self.hidden_size, grid, grid)
        self.register_buffer("pos_embed", pos, persistent=False)

        if learn_context_time_embed:
            self.context_time_embed = nn.Parameter(torch.zeros(1, self.max_context_frames, 1, self.hidden_size))
            nn.init.normal_(self.context_time_embed, std=0.02)
        else:
            self.register_buffer(
                "context_time_embed",
                torch.zeros(1, self.max_context_frames, 1, self.hidden_size),
                persistent=False,
            )

        self.conditioner = CDiTConditioner(
            self.hidden_size,
            action_dim=self.action_dim,
            action_hidden=action_hidden,
            action_seq_depth=action_seq_depth,
            action_seq_heads=action_seq_heads,
            action_seq_dropout=action_seq_dropout,
            max_action_tokens=max_action_tokens,
            context_noise_levels=context_noise_levels,
        )
        self.blocks = nn.ModuleList([
            DiTBlock(self.hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(self.depth)
        ])
        self.final_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(self.hidden_size, self.hidden_size * 2))
        self.head = nn.Linear(self.hidden_size, self.patch_size * self.patch_size * self.latent_channels)
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _encode_context(self, context_latents: torch.Tensor) -> torch.Tensor:
        B, Tc, C, H, W = context_latents.shape
        if H != self.input_size or W != self.input_size:
            raise ValueError(f"Expected context latent size {(self.input_size, self.input_size)}, got {(H, W)}")
        flat = context_latents.view(B * Tc, C, H, W)
        tokens = self.patch_embed(flat)
        tokens = tokens + self.pos_embed.to(device=tokens.device, dtype=tokens.dtype)
        N = tokens.shape[1]
        tokens = tokens.view(B, Tc, N, self.hidden_size)
        time_emb = self.context_time_embed[:, :Tc].to(device=tokens.device, dtype=tokens.dtype)
        tokens = tokens + time_emb
        return tokens.view(B, Tc * N, self.hidden_size)

    def forward(
        self,
        noisy_target: torch.Tensor,
        context_latents: torch.Tensor,
        diffusion_t: torch.Tensor,
        horizons: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        context_noise_bucket: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C, H, W = noisy_target.shape
        if H != self.input_size or W != self.input_size:
            raise ValueError(f"Expected target latent size {(self.input_size, self.input_size)}, got {(H, W)}")
        if C != self.latent_channels:
            raise ValueError(f"Expected latent_channels={self.latent_channels}, got C={C}")

        x = self.patch_embed(noisy_target)
        x = x + self.pos_embed.to(device=x.device, dtype=x.dtype)
        context = self._encode_context(context_latents)

        if self.action_dim <= 0:
            actions = None
            action_mask = None
        elif actions is None:
            actions = noisy_target.new_zeros((B, 1, self.action_dim))
            action_mask = noisy_target.new_ones((B, 1), dtype=torch.bool)

        if self.context_noise_levels > 0 and context_noise_bucket is None:
            context_noise_bucket = diffusion_t.new_zeros((B,))

        cond = self.conditioner(
            diffusion_t=diffusion_t,
            horizons=horizons,
            actions=actions,
            action_mask=action_mask,
            context_noise_bucket=context_noise_bucket,
        )
        for block in self.blocks:
            x = block(x, context, cond)

        shift, scale = self.final_adaLN(cond).chunk(2, dim=-1)
        x = modulate(self.final_norm(x), shift, scale)
        x = self.head(x)
        return unpatchify_2d(x, out_channels=self.latent_channels, patch_size=self.patch_size, out_h=H, out_w=W)
