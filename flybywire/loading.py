from __future__ import annotations
from typing import Optional
from safetensors.torch import load_file
from dit.loading import ModelSource
from .critic import LatentHazardCriticConfig, LatentActionHazardCritic


def load_critic(root: Optional[str] = None, *, device="cpu"):
    src = ModelSource(root)
    cfg = src.config("steering/config.json")
    cc = LatentHazardCriticConfig(
        latent_shape=tuple(int(x) for x in cfg["latent_shape"]),
        action_dim=int(cfg["action_dim"]), context_frames=int(cfg["context_frames"]),
        max_action_tokens=int(cfg["max_action_tokens"]),
        latent_feature_dim=int(cfg["latent_feature_dim"]),
        action_feature_dim=int(cfg["action_feature_dim"]), hidden_dim=int(cfg["hidden_dim"]),
        horizons=tuple(int(h) for h in cfg["horizons"]),
        use_predicted_latent=bool(cfg.get("use_predicted_latent", True)),
        use_residual_latent=bool(cfg.get("use_residual_latent", False)),
        use_tactical_heads=bool(cfg.get("use_tactical_heads", True)),
        use_tactical_value_head=bool(cfg.get("use_tactical_value_head", True)),
        use_tactical_advantage_head=bool(cfg.get("use_tactical_advantage_head", True)),
    )
    model = LatentActionHazardCritic(cc)
    state = load_file(str(src.resolve("steering/model.safetensors")), device="cpu")
    if state and all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: value for k, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False), cfg
