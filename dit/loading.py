from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file
from .vae_model import VAE
from .dit_model import ActionConditionedDiT
from .diffusion_utils import GaussianDiffusion2D
from .world import WorldModel

class ModelSource:
    def __init__(self, root: Optional[str]):
        self.root = str(root or os.environ.get("MODEL_ROOT", "")).strip()
        if not self.root:
            raise ValueError("Provide --model-root or MODEL_ROOT (local HF directory or owner/repo)")
        path = Path(self.root).expanduser()
        self.directory = path.resolve() if path.is_dir() else None
        if self.directory is None and (self.root.startswith(".") or self.root.startswith("/") or self.root.startswith("~")):
            raise FileNotFoundError(f"Local MODEL_ROOT directory does not exist: {self.root}")

    def resolve(self, filename: str) -> Path:
        filename = str(filename)
        supplied = Path(filename).expanduser()
        if supplied.is_file():
            return supplied.resolve()
        if supplied.is_absolute():
            raise FileNotFoundError(supplied)
        if self.directory is not None:
            target = self.directory / filename
            if not target.is_file():
                raise FileNotFoundError(target)
            return target
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(repo_id=self.root, filename=filename, repo_type="model"))

    def config(self, filename: str) -> dict:
        return json.loads(self.resolve(filename).read_text())


def load_weights(model: torch.nn.Module, file: Path) -> None:
    state = load_file(str(file), device="cpu")
    if state and all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)


def load_world(root: Optional[str] = None, *, device="cpu") -> WorldModel:
    src = ModelSource(root)
    vc = src.config("vae/config.json")
    dc = src.config("dit/config.json")
    df = dict(dc.get("diffusion", {}))
    defaults = dict(dc.get("inference_defaults", {}))
    if dc.get("weights_variant", "ema") != "ema":
        print("[note] DiT weights_variant is not 'ema'; loading weights as supplied")
    vae = VAE(in_channels=int(vc["in_channels"]), z_dim=int(vc["z_dim"]),
              dim=int(vc["dim"]), dim_mult=tuple(int(x) for x in vc["dim_mult"]),
              num_res_blocks=int(vc["num_res_blocks"]), dropout=float(vc.get("dropout", 0.0)))
    dit = ActionConditionedDiT(
        latent_channels=int(dc["latent_channels"]), action_dim=int(dc["action_dim"]),
        input_size=int(dc["input_size"]), patch_size=int(dc["patch_size"]),
        hidden_size=int(dc["hidden_size"]), depth=int(dc["depth"]),
        num_heads=int(dc["num_heads"]), mlp_ratio=float(dc.get("mlp_ratio", 4.0)),
        dropout=float(dc.get("dropout", 0.0)),
        max_context_frames=int(dc["max_context_frames"]),
        learn_context_time_embed=bool(dc.get("learn_context_time_embed", True)),
        action_hidden=int(dc["action_hidden"]), action_seq_depth=int(dc["action_seq_depth"]),
        action_seq_heads=int(dc["action_seq_heads"]),
        action_seq_dropout=float(dc.get("action_seq_dropout", 0.0)),
        max_action_tokens=int(dc["max_action_tokens"]),
        context_noise_levels=int(dc.get("context_noise_levels", 8)),
    )
    load_weights(vae, src.resolve("vae/model.safetensors"))
    load_weights(dit, src.resolve("dit/model.safetensors"))
    diffusion = GaussianDiffusion2D(
        num_steps=int(df["num_train_timesteps"]), beta_schedule=str(df["beta_schedule"])
    ).to(torch.device(device))
    diffusion_cfg = {**df, **defaults}
    return WorldModel(vae.to(device).eval().requires_grad_(False),
                      dit.to(device).eval().requires_grad_(False), diffusion,
                      device=device, vae_config=vc, dit_config=dc,
                      diffusion_config=diffusion_cfg, resolve=src.resolve).eval()
