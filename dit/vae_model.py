from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(32, in_channels), num_channels=in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.skip(x)


class Downsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Encoder2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_dim: int,
        z_dim: int,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        dims = [base_dim * m for m in dim_mult]
        self.conv_in = nn.Conv2d(in_channels, dims[0], 3, padding=1)

        blocks = []
        curr = dims[0]
        for i, dim in enumerate(dims):
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock2D(curr, dim, dropout=dropout))
                curr = dim
            if i < len(dims) - 1:
                blocks.append(Downsample2D(curr))

        self.blocks = nn.Sequential(*blocks)
        self.norm = nn.GroupNorm(num_groups=min(32, curr), num_channels=curr)
        self.act = nn.SiLU()
        self.conv_out = nn.Conv2d(curr, z_dim * 2, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.conv_out(self.act(self.norm(x)))
        return x


class Decoder2D(nn.Module):
    def __init__(
        self,
        out_channels: int,
        base_dim: int,
        z_dim: int,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        dims = [base_dim * m for m in dim_mult]
        curr = dims[-1]
        self.conv_in = nn.Conv2d(z_dim, curr, 3, padding=1)

        blocks = []
        for i, dim in reversed(list(enumerate(dims))):
            for _ in range(num_res_blocks + 1):
                blocks.append(ResidualBlock2D(curr, dim, dropout=dropout))
                curr = dim
            if i > 0:
                blocks.append(Upsample2D(curr))

        self.blocks = nn.Sequential(*blocks)
        self.norm = nn.GroupNorm(num_groups=min(32, curr), num_channels=curr)
        self.act = nn.SiLU()
        self.conv_out = nn.Conv2d(curr, out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.conv_in(z)
        z = self.blocks(z)
        return self.conv_out(self.act(self.norm(z)))


class VAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        z_dim: int = 4,
        dim: int = 128,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.z_dim = int(z_dim)
        self.dim_mult = tuple(int(m) for m in dim_mult)
        self.downsample_factor = 2 ** (len(self.dim_mult) - 1)

        self.encoder = Encoder2D(
            in_channels,
            dim,
            z_dim,
            dim_mult=self.dim_mult,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
        )
        self.decoder = Decoder2D(
            in_channels,
            dim,
            z_dim,
            dim_mult=self.dim_mult,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
        )

    def _fold_time(self, x_bcthw: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int, int]]:
        B, C, T, H, W = x_bcthw.shape
        x_btchw = x_bcthw.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        return x_btchw, (B, C, T, H, W)

    def _unfold_time(self, x_btchw: torch.Tensor, shape: Tuple[int, int, int, int, int]) -> torch.Tensor:
        B, _C, T, _H, _W = shape
        Cx, Hx, Wx = int(x_btchw.shape[1]), int(x_btchw.shape[2]), int(x_btchw.shape[3])
        return x_btchw.view(B, T, Cx, Hx, Wx).permute(0, 2, 1, 3, 4).contiguous()

    def _pad_to_factor(
        self, x_bcthw: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int, int, int], Tuple[int, int]]:
        _, _, _, H, W = x_bcthw.shape
        factor = int(self.downsample_factor)
        pad_h = (factor - (H % factor)) % factor
        pad_w = (factor - (W % factor)) % factor
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        if top or bottom or left or right:
            x_bcthw = F.pad(x_bcthw, (left, right, top, bottom))
        return x_bcthw, (left, right, top, bottom), (H, W)

    @staticmethod
    def _crop_to_size(
        x_bcthw: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
        pad: Optional[Tuple[int, int, int, int]] = None,
    ) -> torch.Tensor:
        if pad is not None:
            left, right, top, bottom = pad
            h_end = x_bcthw.shape[-2] - bottom if bottom > 0 else x_bcthw.shape[-2]
            w_end = x_bcthw.shape[-1] - right if right > 0 else x_bcthw.shape[-1]
            x_bcthw = x_bcthw[..., top:h_end, left:w_end]

        if output_size is not None:
            target_h, target_w = output_size
            x_bcthw = x_bcthw[..., :target_h, :target_w]
        return x_bcthw.contiguous()

    def encode(self, x_bcthw: torch.Tensor, scale=(0.0, 1.0)):
        x_bcthw, _pad, _orig_hw = self._pad_to_factor(x_bcthw)
        x_btchw, shape = self._fold_time(x_bcthw)
        h = self.encoder(x_btchw)
        mu, logvar = h.chunk(2, dim=1)
        s0, s1 = scale
        mu = (mu - float(s0)) * float(s1)
        return self._unfold_time(mu, shape), self._unfold_time(logvar, shape)

    def decode(
        self,
        z_bcthw: torch.Tensor,
        scale=(0.0, 1.0),
        output_size: Optional[Tuple[int, int]] = None,
    ):
        z_btchw, shape = self._fold_time(z_bcthw)
        s0, s1 = scale
        z_btchw = z_btchw / float(s1) + float(s0)
        x = self.decoder(z_btchw)
        x = self._unfold_time(x, shape)
        if output_size is not None:
            x = self._crop_to_size(x, output_size=output_size)
        return x

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_bcthw: torch.Tensor, scale=(0.0, 1.0), sample_latent: bool = False):
        x_pad, pad, orig_hw = self._pad_to_factor(x_bcthw)
        mu, logvar = self.encode(x_pad, scale=scale)
        z = self.reparameterize(mu, logvar) if sample_latent else mu
        recon = self.decode(z, scale=scale)
        recon = self._crop_to_size(recon, output_size=orig_hw, pad=pad)
        return recon, mu, logvar
