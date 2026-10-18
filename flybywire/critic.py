from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Optional as _Optional
import torch
import torch.nn.functional as F
from torch import nn

@dataclass
class LatentHazardCriticConfig:
    latent_shape: tuple[int, int, int]  # [Z,h,w]
    action_dim: int = 6
    context_frames: int = 4
    max_action_tokens: int = 32
    latent_feature_dim: int = 256
    action_feature_dim: int = 128
    hidden_dim: int = 512
    horizons: tuple[int, ...] = (4, 8, 16, 32)
    use_predicted_latent: bool = True
    use_residual_latent: bool = False
    use_tactical_heads: bool = True
    use_tactical_value_head: bool = True
    use_tactical_advantage_head: bool = True


class LatentFrameEncoder(nn.Module):
    def __init__(self, latent_shape: Sequence[int], out_dim: int = 256):
        super().__init__()
        z, h, w = [int(x) for x in latent_shape]
        width = max(32, min(128, int(out_dim) // 2))
        self.net = nn.Sequential(
            nn.Conv2d(z, width, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(width * 4 * 4, int(out_dim)),
            nn.SiLU(),
        )
        self.latent_shape = (z, h, w)
        self.out_dim = int(out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 4:
            raise ValueError(f"Expected latent frame [B,Z,h,w], got {tuple(z.shape)}")
        return self.net(z.float())


class LatentActionHazardCritic(nn.Module):
    """Hazard/action critic over the same latent substrate used by the DiT.
      context_latents:  [B,Tc,Z,h,w]
      actions:          [B,L,A]
      predicted_latent: optional [B,Z,h,w],
      residual_latent:  optional [B,Z,h,w],
    """
    def __init__(self, cfg: LatentHazardCriticConfig):
        super().__init__()
        self.cfg = cfg
        self.horizons = tuple(int(h) for h in cfg.horizons)
        self.latent_shape = tuple(int(x) for x in cfg.latent_shape)
        self.context_frames = int(cfg.context_frames)
        self.action_dim = int(cfg.action_dim)
        self.use_predicted_latent = bool(cfg.use_predicted_latent)
        self.use_residual_latent = bool(cfg.use_residual_latent)
        self.use_tactical_heads = bool(getattr(cfg, "use_tactical_heads", True))
        self.use_tactical_value_head = bool(getattr(cfg, "use_tactical_value_head", True))
        self.use_tactical_advantage_head = bool(getattr(cfg, "use_tactical_advantage_head", True))

        self.frame_encoder = LatentFrameEncoder(self.latent_shape, out_dim=int(cfg.latent_feature_dim))
        self.context_gru = nn.GRU(
            input_size=int(cfg.latent_feature_dim),
            hidden_size=int(cfg.latent_feature_dim),
            batch_first=True,
        )
        self.action_proj = nn.Linear(self.action_dim, int(cfg.action_feature_dim))
        self.action_gru = nn.GRU(
            input_size=int(cfg.action_feature_dim),
            hidden_size=int(cfg.action_feature_dim),
            batch_first=True,
        )
        extra = int(cfg.latent_feature_dim) + int(cfg.action_feature_dim)
        if self.use_predicted_latent:
            extra += int(cfg.latent_feature_dim)
        if self.use_residual_latent:
            extra += int(cfg.latent_feature_dim)
        self.trunk = nn.Sequential(
            nn.Linear(extra, int(cfg.hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(cfg.hidden_dim), int(cfg.hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(cfg.hidden_dim), int(cfg.hidden_dim)),
            nn.SiLU(),
        )
        self.death_head = nn.Linear(int(cfg.hidden_dim), len(self.horizons))
        self.health_loss_head = nn.Linear(int(cfg.hidden_dim), len(self.horizons))
        self.damage_head = nn.Linear(int(cfg.hidden_dim), len(self.horizons))
        self.frag_gain_head = nn.Linear(int(cfg.hidden_dim), len(self.horizons))
        self.damage_dealt_head = nn.Linear(int(cfg.hidden_dim), len(self.horizons))
        self.reward_head = nn.Linear(int(cfg.hidden_dim), len(self.horizons))
        self.tactical_value_head = nn.Linear(int(cfg.hidden_dim), len(self.horizons))
        if self.use_tactical_advantage_head:
            self.tactical_advantage_head = nn.Linear(int(cfg.hidden_dim), len(self.horizons))

    def encode_context(self, context_latents: torch.Tensor) -> torch.Tensor:
        if context_latents.ndim != 5:
            raise ValueError(f"Expected context_latents [B,Tc,Z,h,w], got {tuple(context_latents.shape)}")
        b, tc = int(context_latents.shape[0]), int(context_latents.shape[1])
        flat = context_latents.reshape(b * tc, *context_latents.shape[2:])
        tokens = self.frame_encoder(flat).reshape(b, tc, -1)
        _out, h = self.context_gru(tokens)
        return h[-1]

    def encode_actions(self, actions: torch.Tensor, action_mask: _Optional[torch.Tensor] = None) -> torch.Tensor:
        if actions.ndim != 3:
            raise ValueError(f"Expected actions [B,L,A], got {tuple(actions.shape)}")
        x = self.action_proj(actions.float())
        if action_mask is not None:
            x = x * action_mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
        _out, h = self.action_gru(x)
        return h[-1]

    def forward(
        self,
        context_latents: torch.Tensor,
        actions: torch.Tensor,
        *,
        action_mask: _Optional[torch.Tensor] = None,
        predicted_latent: _Optional[torch.Tensor] = None,
        residual_latent: _Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        features = [self.encode_context(context_latents), self.encode_actions(actions, action_mask=action_mask)]
        if self.use_predicted_latent:
            if predicted_latent is None:
                predicted_latent = torch.zeros((context_latents.shape[0], *self.latent_shape), device=context_latents.device, dtype=context_latents.dtype)
            features.append(self.frame_encoder(predicted_latent))
        if self.use_residual_latent:
            if residual_latent is None:
                residual_latent = torch.zeros((context_latents.shape[0], *self.latent_shape), device=context_latents.device, dtype=context_latents.dtype)
            features.append(self.frame_encoder(residual_latent))
        h = self.trunk(torch.cat(features, dim=-1))
        damage_taken = F.softplus(self.damage_head(h))
        frag_gain = F.softplus(self.frag_gain_head(h))
        damage_dealt = F.softplus(self.damage_dealt_head(h))
        reward = self.reward_head(h)
        tactical_value = self.tactical_value_head(h)
        tactical_advantage = self.tactical_advantage_head(h) if self.use_tactical_advantage_head else None
        result = {
            "death_logits": self.death_head(h),
            "health_loss": F.softplus(self.health_loss_head(h)),
            "damage": damage_taken,
            "damage_taken": damage_taken,
            "frag_gain": frag_gain,
            "damage_dealt": damage_dealt,
            "reward": reward,
            "tactical_value": tactical_value,
        }
        if tactical_advantage is not None:
            result["tactical_advantage"] = tactical_advantage
        return result

    def score_death_risk(
        self,
        context_latents: torch.Tensor,
        actions: torch.Tensor,
        *,
        action_mask: _Optional[torch.Tensor] = None,
        predicted_latent: _Optional[torch.Tensor] = None,
        residual_latent: _Optional[torch.Tensor] = None,
        horizon_index: int = -1,
    ) -> torch.Tensor:
        out = self.forward(
            context_latents,
            actions,
            action_mask=action_mask,
            predicted_latent=predicted_latent,
            residual_latent=residual_latent,
        )
        return torch.sigmoid(out["death_logits"][..., int(horizon_index)])

    def score_tactical_badness(
        self,
        context_latents: torch.Tensor,
        actions: torch.Tensor,
        *,
        action_mask: _Optional[torch.Tensor] = None,
        predicted_latent: _Optional[torch.Tensor] = None,
        residual_latent: _Optional[torch.Tensor] = None,
        horizon_index: int = -1,
        death_weight: float = 1.0,
        health_loss_weight: float = 0.01,
        damage_taken_weight: float = 0.01,
        frag_gain_weight: float = 1.0,
        damage_dealt_weight: float = 0.02,
        reward_weight: float = 0.0,
        use_tactical_value: bool = False,
        use_tactical_advantage: bool = False,
    ) -> torch.Tensor:
        out = self.forward(
            context_latents,
            actions,
            action_mask=action_mask,
            predicted_latent=predicted_latent,
            residual_latent=residual_latent,
        )
        i = int(horizon_index)
        if bool(use_tactical_advantage) and "tactical_advantage" in out:
            return -out["tactical_advantage"][..., i]
        if bool(use_tactical_value) and "tactical_value" in out:
            return -out["tactical_value"][..., i]
        return (
            float(death_weight) * torch.sigmoid(out["death_logits"])[..., i]
            + float(health_loss_weight) * out["health_loss"][..., i]
            + float(damage_taken_weight) * out.get("damage_taken", out["damage"])[..., i]
            - float(frag_gain_weight) * out.get("frag_gain", torch.zeros_like(out["health_loss"]))[..., i]
            - float(damage_dealt_weight) * out.get("damage_dealt", torch.zeros_like(out["health_loss"]))[..., i]
            - float(reward_weight) * out.get("reward", torch.zeros_like(out["health_loss"]))[..., i]
        )

    def export_config(self) -> dict:
        return {
            "class": "LatentActionHazardCritic",
            "latent_shape": list(self.latent_shape),
            "action_dim": int(self.action_dim),
            "context_frames": int(self.context_frames),
            "max_action_tokens": int(self.cfg.max_action_tokens),
            "latent_feature_dim": int(self.cfg.latent_feature_dim),
            "action_feature_dim": int(self.cfg.action_feature_dim),
            "hidden_dim": int(self.cfg.hidden_dim),
            "horizons": list(self.horizons),
            "use_predicted_latent": bool(self.use_predicted_latent),
            "use_residual_latent": bool(self.use_residual_latent),
            "use_tactical_heads": bool(self.use_tactical_heads),
            "use_tactical_value_head": bool(self.use_tactical_value_head),
            "use_tactical_advantage_head": bool(self.use_tactical_advantage_head),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: dict) -> "LatentActionHazardCritic":
        cfg_obj = checkpoint.get("model_config") or checkpoint.get("config")
        if cfg_obj is None:
            raise ValueError("Checkpoint missing model_config/config")
        cfg = LatentHazardCriticConfig(
            latent_shape=tuple(int(x) for x in cfg_obj["latent_shape"]),
            action_dim=int(cfg_obj.get("action_dim", 6)),
            context_frames=int(cfg_obj.get("context_frames", 4)),
            max_action_tokens=int(cfg_obj.get("max_action_tokens", 32)),
            latent_feature_dim=int(cfg_obj.get("latent_feature_dim", 256)),
            action_feature_dim=int(cfg_obj.get("action_feature_dim", 128)),
            hidden_dim=int(cfg_obj.get("hidden_dim", 512)),
            horizons=tuple(int(x) for x in cfg_obj.get("horizons", (4, 8, 16, 32))),
            use_predicted_latent=bool(cfg_obj.get("use_predicted_latent", True)),
            use_residual_latent=bool(cfg_obj.get("use_residual_latent", False)),
            use_tactical_heads=bool(cfg_obj.get("use_tactical_heads", True)),
            use_tactical_value_head=bool(cfg_obj.get("use_tactical_value_head", True)),
            use_tactical_advantage_head=bool(cfg_obj.get("use_tactical_advantage_head", True)),
        )
        model = cls(cfg)
        state = checkpoint.get("model", checkpoint.get("state_dict"))
        if state is None:
            raise ValueError("Checkpoint missing model/state_dict")
        model.load_state_dict(state, strict=False)
        return model
