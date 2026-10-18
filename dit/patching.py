from __future__ import annotations

import math

import torch
import torch.nn as nn


class PatchEmbed2D(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 2):
        super().__init__()
        self.in_channels = int(in_channels)
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.proj = nn.Linear(self.in_channels * self.patch_size * self.patch_size, self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        if H % p != 0 or W % p != 0:
            raise ValueError(f"Spatial dims must be divisible by patch_size={p}, got {(H, W)}")
        x = x.view(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(B, (H // p) * (W // p), C * p * p)
        return self.proj(x)


def unpatchify_2d(tokens: torch.Tensor, out_channels: int, patch_size: int, out_h: int, out_w: int) -> torch.Tensor:
    B, N, D = tokens.shape
    p = int(patch_size)
    gh = out_h // p
    gw = out_w // p
    if N != gh * gw:
        raise ValueError(f"Token count mismatch: N={N} vs grid={(gh, gw)}")
    x = tokens.view(B, gh, gw, out_channels, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.view(B, out_channels, out_h, out_w)


def build_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int, device=None, dtype=torch.float32) -> torch.Tensor:
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim must be divisible by 4 for 2D sin/cos, got {embed_dim}")

    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_h, device=device, dtype=dtype),
        torch.arange(grid_w, device=device, dtype=dtype),
        indexing="ij",
    )
    grid_y = grid_y.reshape(-1)
    grid_x = grid_x.reshape(-1)

    half = embed_dim // 2
    omega = torch.arange(half // 2, device=device, dtype=dtype)
    omega = 1.0 / (10000 ** (omega / max(1, (half // 2) - 1)))

    out_y = torch.einsum("n,d->nd", grid_y, omega)
    out_x = torch.einsum("n,d->nd", grid_x, omega)

    pos = torch.cat([out_y.sin(), out_y.cos(), out_x.sin(), out_x.cos()], dim=-1)
    return pos.unsqueeze(0)
