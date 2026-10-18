from __future__ import annotations

import argparse
import json
import platform
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from dit.bootstrap import load_bootstrap_segment
from dit.hud import draw_hud
from dit.loading import load_world
from dit.diffusion_utils import GaussianDiffusion2D
from dit.vae_model import VAE


BUTTONS: List[str] = [
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "ATTACK",
    # "SPEED",
    # "USE",
    "TURN_LEFT_RIGHT_DELTA",
    # "LOOK_UP_DOWN_DELTA",
    # "JUMP",
]
BUTTON_INDEX: Dict[str, int] = {name: i for i, name in enumerate(BUTTONS)}
BUTTON_MAX_VALUES: Dict[str, float] = {
    "TURN_LEFT_RIGHT_DELTA": 180.0,
    # "LOOK_UP_DOWN_DELTA": 90.0,
}

ARROW_LEFT = 2424832
ARROW_UP = 2490368
ARROW_RIGHT = 2555904
ARROW_DOWN = 2621440
NO_KEY_CODES = {-1, 255, 0xFFFFFFFF}


@dataclass
class InputState:
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0
    left_down: bool = False
    right_down: bool = False
    middle_down: bool = False
    last_mouse_pos: Optional[Tuple[int, int]] = None

    def reset_motion(self) -> Tuple[float, float]:
        dx, dy = float(self.mouse_dx), float(self.mouse_dy)
        self.mouse_dx = 0.0
        self.mouse_dy = 0.0
        return dx, dy


INPUT_STATE = InputState()


def detect_device(device_name: str) -> tuple[torch.device, str]:
    if torch.cuda.is_available():
        return torch.device(device_name), device_name
    if platform.system() == "Darwin" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clamp_action_vector(vec: Sequence[float]) -> List[float]:
    if len(vec) != len(BUTTONS):
        raise ValueError(f"Expected {len(BUTTONS)} action values, got {len(vec)}")

    out: List[float] = []
    for name, value in zip(BUTTONS, vec):
        value = float(value)
        if name in BUTTON_MAX_VALUES:
            limit = float(BUTTON_MAX_VALUES[name])
            value = max(-limit, min(limit, value))
        else:
            value = max(0.0, min(1.0, value))
        out.append(value)
    return out


def choose_bootstrap_index(dataset_len: int, requested: Optional[int], seed: int) -> int:
    if dataset_len <= 0:
        raise ValueError("Dataset is empty.")
    max_idx = min(99, dataset_len - 1)
    if requested is None:
        return random.Random(seed + 17).randint(0, max_idx)
    return max(0, min(int(requested), dataset_len - 1))


def pad_or_trim_actions(action_tensor: Optional[torch.Tensor], action_dim: int) -> Optional[torch.Tensor]:
    if action_tensor is None:
        return None
    x = torch.as_tensor(action_tensor).float()
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2:
        raise ValueError(f"Expected actions with shape [T,A], got {tuple(x.shape)}")
    if x.shape[-1] == action_dim:
        return x.contiguous()
    out = torch.zeros((x.shape[0], action_dim), dtype=torch.float32)
    n = min(int(x.shape[-1]), int(action_dim))
    if n > 0:
        out[:, :n] = x[:, :n]
    return out.contiguous()


def build_bootstrap_segment(
    npz_root: str,
    context_frames: int,
    horizon: int,
    start_index: Optional[int],
    seed: int,
) -> tuple[int, torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    return load_bootstrap_segment(
        source=npz_root, context_frames=context_frames, horizon=horizon,
        index=start_index, seed=seed, action_dim=len(BUTTONS),
    )


@torch.no_grad()
def encode_obs_like_training(
    vae: VAE,
    obs_btchw: torch.Tensor,
    sample_latent: bool = False,
) -> torch.Tensor:
    x_bcthw = obs_btchw.permute(0, 2, 1, 3, 4).contiguous()
    x_bcthw = x_bcthw * 2.0 - 1.0
    mu, logvar = vae.encode(x_bcthw, scale=(0.0, 1.0))
    z_bcthw = vae.reparameterize(mu, logvar) if sample_latent else mu
    return z_bcthw.permute(0, 2, 1, 3, 4).contiguous()


@torch.no_grad()
def decode_latent_like_training(
    vae: VAE,
    z_bzhw: torch.Tensor,
    output_size: tuple[int, int],
) -> torch.Tensor:
    x_bcthw = vae.decode(z_bzhw.unsqueeze(2), scale=(0.0, 1.0), output_size=output_size)
    x_bchw = x_bcthw.squeeze(2)
    return (x_bchw * 0.5 + 0.5).clamp(0.0, 1.0)


def tensor_to_uint8_rgb(img_bchw: torch.Tensor) -> np.ndarray:
    img = img_bchw[0].detach().cpu().permute(1, 2, 0).numpy()
    return (img * 255.0).round().clip(0, 255).astype(np.uint8)


def compute_display_size(frame_hw: Tuple[int, int], display_width: int, display_height: int, display_scale: int) -> Tuple[int, int]:
    frame_h, frame_w = int(frame_hw[0]), int(frame_hw[1])
    scale = max(1, int(display_scale))
    if scale > 1:
        return frame_w * scale, frame_h * scale
    if int(display_width) > 0 and int(display_height) > 0:
        return int(display_width), int(display_height)
    return frame_w, frame_h


def action_history_to_tensors(
    history: Sequence[Sequence[float]],
    required_len: int,
    action_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    actions = torch.zeros(1, required_len, action_dim, dtype=torch.float32, device=device)
    mask = torch.zeros(1, required_len, dtype=torch.bool, device=device)
    history_tail = list(history)[-required_len:]
    offset = max(0, required_len - len(history_tail))
    for i, vec in enumerate(history_tail):
        actions[0, offset + i] = torch.tensor(vec, dtype=torch.float32, device=device)
        mask[0, offset + i] = True
    return actions, mask


def mouse_callback(event: int, x: int, y: int, flags: int, param: Any) -> None:
    del flags, param
    global INPUT_STATE

    if INPUT_STATE.last_mouse_pos is None:
        INPUT_STATE.last_mouse_pos = (x, y)

    last_x, last_y = INPUT_STATE.last_mouse_pos
    if event == cv2.EVENT_MOUSEMOVE:
        INPUT_STATE.mouse_dx += float(x - last_x)
        INPUT_STATE.mouse_dy += float(y - last_y)
        INPUT_STATE.last_mouse_pos = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        INPUT_STATE.left_down = True
        INPUT_STATE.last_mouse_pos = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        INPUT_STATE.left_down = False
        INPUT_STATE.last_mouse_pos = (x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        INPUT_STATE.right_down = True
        INPUT_STATE.last_mouse_pos = (x, y)
    elif event == cv2.EVENT_RBUTTONUP:
        INPUT_STATE.right_down = False
        INPUT_STATE.last_mouse_pos = (x, y)
    elif event == cv2.EVENT_MBUTTONDOWN:
        INPUT_STATE.middle_down = True
        INPUT_STATE.last_mouse_pos = (x, y)
    elif event == cv2.EVENT_MBUTTONUP:
        INPUT_STATE.middle_down = False
        INPUT_STATE.last_mouse_pos = (x, y)


def normalize_key_code(key_code: int) -> int:
    return int(key_code & 0xFFFFFFFF)


def key_is_active(active_keys: Dict[int, float], now: float, hold_s: float, candidates: Sequence[int]) -> bool:
    for code in candidates:
        code = normalize_key_code(code)
        last_seen = active_keys.get(code)
        if last_seen is not None and (now - last_seen) <= hold_s:
            return True
    return False


def update_active_keys(active_keys: Dict[int, float], key_code: int, now: float, hold_s: float) -> None:
    norm = normalize_key_code(key_code)
    if norm not in NO_KEY_CODES:
        active_keys[norm] = now

    stale_before = now - (hold_s * 4.0)
    stale = [code for code, ts in active_keys.items() if ts < stale_before]
    for code in stale:
        active_keys.pop(code, None)


def make_action_from_input(
    active_keys: Dict[int, float],
    now: float,
    hold_s: float,
    keyboard_turn_step: float,
    keyboard_look_step: float,
    mouse_turn_scale: float,
    mouse_look_scale: float,
    invert_mouse_y: bool,
) -> tuple[List[float], bool]:
    global INPUT_STATE
    vec = [0.0] * len(BUTTONS)

    should_quit = key_is_active(active_keys, now, hold_s, [27, ord("q"), ord("Q")])

    if key_is_active(active_keys, now, hold_s, [ord("w"), ord("W"), ARROW_UP]):
        vec[BUTTON_INDEX["MOVE_FORWARD"]] = 1.0
    if key_is_active(active_keys, now, hold_s, [ord("s"), ord("S"), ARROW_DOWN]):
        vec[BUTTON_INDEX["MOVE_BACKWARD"]] = 1.0
    if key_is_active(active_keys, now, hold_s, [ord("a"), ord("A")]):
        vec[BUTTON_INDEX["MOVE_LEFT"]] = 1.0
    if key_is_active(active_keys, now, hold_s, [ord("d"), ord("D")]):
        vec[BUTTON_INDEX["MOVE_RIGHT"]] = 1.0
    if key_is_active(active_keys, now, hold_s, [ord("f"), ord("F")]) or INPUT_STATE.left_down:
        vec[BUTTON_INDEX["ATTACK"]] = 1.0

    turn = 0.0
    look = 0.0


    mouse_dx, mouse_dy = INPUT_STATE.reset_motion()
    turn += float(mouse_dx) * float(mouse_turn_scale)
    look_delta = float(mouse_dy) * float(mouse_look_scale)
    look += (-look_delta if invert_mouse_y else look_delta)

    vec[BUTTON_INDEX["TURN_LEFT_RIGHT_DELTA"]] = turn
    return clamp_action_vector(vec), should_quit


def ensure_writer(record_path: str, fps: float, size_hw: Tuple[int, int]) -> cv2.VideoWriter:
    path = Path(record_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = int(size_hw[0]), int(size_hw[1])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {record_path!r}")
    return writer


def derive_latent_cache_path(record_path: Optional[str], latent_cache_path: Optional[str]) -> Optional[Path]:
    if latent_cache_path:
        return Path(latent_cache_path)
    if record_path:
        rp = Path(record_path)
        return rp.with_name(f"{rp.stem}_latents.npz")
    return None


def build_ddim_timesteps(num_train_steps: int, num_sample_steps: int) -> List[int]:
    if num_train_steps <= 0:
        raise ValueError("num_train_steps must be positive")
    if num_sample_steps <= 0:
        raise ValueError("num_sample_steps must be positive")
    if num_sample_steps >= num_train_steps:
        return list(range(num_train_steps))

    timesteps = np.linspace(0, num_train_steps - 1, num=num_sample_steps, dtype=np.int64)
    timesteps = np.unique(timesteps)
    if int(timesteps[-1]) != int(num_train_steps - 1):
        timesteps[-1] = int(num_train_steps - 1)
    return [int(x) for x in timesteps.tolist()]


@torch.no_grad()
def ddim_sample_loop(
    diffusion: GaussianDiffusion2D,
    model_fn,
    shape: Tuple[int, ...],
    device: torch.device,
    num_sample_steps: int,
    eta: float = 0.0,
    clip_x0: Optional[Tuple[float, float]] = (-5.0, 5.0),
) -> torch.Tensor:
    xt = torch.randn(shape, device=device)
    schedule = build_ddim_timesteps(diffusion.num_steps, int(num_sample_steps))

    for idx in range(len(schedule) - 1, -1, -1):
        t_int = int(schedule[idx])
        prev_t_int = int(schedule[idx - 1]) if idx > 0 else -1

        t = torch.full((shape[0],), t_int, device=device, dtype=torch.long)
        eps = model_fn(xt, t)
        x0 = diffusion.predict_x0_from_eps(xt, t, eps)
        if clip_x0 is not None:
            x0 = x0.clamp(float(clip_x0[0]), float(clip_x0[1]))

        alpha_t = diffusion.alphas_cumprod[t_int].to(device=device, dtype=xt.dtype)
        alpha_prev = (
            diffusion.alphas_cumprod[prev_t_int].to(device=device, dtype=xt.dtype)
            if prev_t_int >= 0
            else torch.tensor(1.0, device=device, dtype=xt.dtype)
        )

        sigma = float(eta) * torch.sqrt(((1.0 - alpha_prev) / (1.0 - alpha_t)).clamp_min(0.0)) * torch.sqrt((1.0 - (alpha_t / alpha_prev)).clamp_min(0.0))
        pred_dir = torch.sqrt((1.0 - alpha_prev - sigma.square()).clamp_min(0.0)) * eps
        noise = torch.randn_like(xt) if (prev_t_int >= 0 and float(eta) > 0.0) else torch.zeros_like(xt)

        xt = torch.sqrt(alpha_prev) * x0 + pred_dir + sigma * noise

    return xt


def save_rollout_cache(
    out_path: Path,
    generated_latents: List[np.ndarray],
    action_vectors: List[np.ndarray],
    step_latencies_s: List[float],
    bootstrap_obs: torch.Tensor,
    bootstrap_latents: torch.Tensor,
    bootstrap_info: Dict[str, Any],
    start_index: int,
    frame_hw: Tuple[int, int],
    args: argparse.Namespace,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if generated_latents:
        latents = np.stack(generated_latents, axis=0)
        actions = np.stack(action_vectors, axis=0)
        latencies = np.asarray(step_latencies_s, dtype=np.float32)
    else:
        lat_h = int(bootstrap_latents.shape[-2])
        lat_w = int(bootstrap_latents.shape[-1])
        latents = np.zeros((0, args.vae_z_dim, lat_h, lat_w), dtype=np.float16)
        actions = np.zeros((0, len(BUTTONS)), dtype=np.float32)
        latencies = np.zeros((0,), dtype=np.float32)

    bootstrap_obs_u8 = (
        bootstrap_obs.detach().cpu().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).numpy()
    )
    bootstrap_latents_np = bootstrap_latents.detach().cpu().to(torch.float16).numpy()

    meta = {
        "start_index": int(start_index),
        "seed": int(args.seed),
        "context_frames": int(args.context_frames),
        "horizon": int(args.horizon),
        "diffusion_steps": int(args.diffusion_steps),
        "beta_schedule": str(args.beta_schedule),
        "sampler": str(args.sampler),
        "ddim_steps": int(args.ddim_steps),
        "ddim_eta": float(args.ddim_eta),
        "context_noise_bucket": int(args.context_noise_bucket),
        "context_noise_levels": int(args.context_noise_levels),
        "action_hidden": int(args.action_hidden),
        "action_seq_depth": int(args.action_seq_depth),
        "action_seq_heads": int(args.action_seq_heads),
        "max_action_tokens": int(args.max_action_tokens),
        "mouse_turn_scale": float(args.mouse_turn_scale),
        "mouse_look_scale": float(args.mouse_look_scale),
        "keyboard_turn_step": float(args.keyboard_turn_step),
        "keyboard_look_step": float(args.keyboard_look_step),
        "invert_mouse_y": bool(args.invert_mouse_y),
        "frame_hw": [int(frame_hw[0]), int(frame_hw[1])],
        "button_order": BUTTONS,
        "button_max_values": BUTTON_MAX_VALUES,
        "bootstrap_info": bootstrap_info,
    }
    meta_json = json.dumps(meta, ensure_ascii=False, default=str)

    np.savez_compressed(
        out_path,
        generated_latents=latents,
        actions=actions,
        step_latency_s=latencies,
        bootstrap_obs=bootstrap_obs_u8,
        bootstrap_latents=bootstrap_latents_np,
        button_order=np.asarray(BUTTONS),
        frame_hw=np.asarray(frame_hw, dtype=np.int32),
        meta_json=np.asarray(meta_json),
    )

    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Live autoregressive DiT rollout using HF safetensors + Parquet bootstrap."
    )

    p.add_argument("--model-root", default=None,
                   help="Local HF model directory or org/repo ID; defaults to MODEL_ROOT.")
    p.add_argument("--bootstrap-data", dest="npz_root", default=None,
                   help="Parquet bootstrap file, local path or repo-relative; defaults to bootstrap/bootstrap.parquet.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action="store_true")

    p.add_argument("--horizon", type=int, default=1,
                   help="Visible-frame horizon; currently fixed to 1.")
    p.add_argument("--context_noise_bucket", type=int, default=0,
                   help="Context-noise conditioning bucket to use at inference. 0 is the clean/default setting.")

    p.add_argument("--sampler", type=str, default="ddim", choices=["ddim", "ddpm"],
                   help="Inference sampler. DDIM is the default for faster rollout.")
    p.add_argument("--ddim_steps", type=int, default=50,
                   help="Number of DDIM sampling steps to use when --sampler ddim.")
    p.add_argument("--ddim_eta", type=float, default=0.0,
                   help="DDIM eta. 0.0 gives deterministic DDIM; >0 adds stochasticity.")
    p.add_argument("--sample_latent", action="store_true")

    p.add_argument("--start_index", type=int, default=None,
                   help="Parquet bootstrap row index; defaults to deterministic random selection.")
    p.add_argument("--max_steps", type=int, default=0, help="0 means run until Q/Esc.")
    p.add_argument("--display_scale", type=int, default=1)
    p.add_argument("--display_width", type=int, default=1280,
                   help="Final display/record width. Set <= 0 to use display_scale instead.")
    p.add_argument("--display_height", type=int, default=960,
                   help="Final display/record height. Set <= 0 to use display_scale instead.")
    p.add_argument("--frame_delay_ms", type=int, default=1)
    p.add_argument("--hud", choices=("compact", "off"), default="compact")
    p.add_argument("--record_path", type=str, default=None,
                   help="Optional MP4 path for the rendered rollout.")
    p.add_argument("--latent_cache_path", type=str, default=None,
                   help="Optional NPZ path for generated latents/actions. Defaults to <record_path stem>_latents.npz when record_path is set.")
    p.add_argument("--seed_actions_from_bootstrap", action=argparse.BooleanOptionalAction, default=True,
                   help="Initialize the rolling action history from the sampled bootstrap segment actions.")

    p.add_argument("--mouse_turn_scale", type=float, default=1.0)
    p.add_argument("--mouse_look_scale", type=float, default=0.5)
    p.add_argument("--keyboard_turn_step", type=float, default=6.0)
    p.add_argument("--keyboard_look_step", type=float, default=4.0)
    p.add_argument("--key_hold_ms", type=float, default=140.0,
                   help="Approximate hold time for cv2 keyboard events so WASD feels continuous.")
    p.add_argument("--invert_mouse_y", action="store_true")
    p.add_argument("--window_name", type=str, default="Live DiT Rollout")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if str(args.sampler) == "ddim" and int(args.ddim_steps) <= 0:
        raise ValueError("ddim_steps must be positive")

    set_seed(args.seed)
    device, device_kind = detect_device(args.device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    world = load_world(args.model_root, device=device)
    vae, model, diffusion = world.vae, world.dit, world.diffusion
    cfg = world.dit_config
    args.action_dim = int(cfg["action_dim"])
    args.context_frames = int(cfg["max_context_frames"])
    args.vae_z_dim = int(world.vae_config["z_dim"])
    args.latent_size = int(cfg["input_size"])
    args.context_noise_levels = int(cfg.get("context_noise_levels", 8))
    if not 0 <= int(args.context_noise_bucket) < max(1, args.context_noise_levels):
        raise ValueError("context_noise_bucket out of range")
    args.diffusion_steps = int(world.diffusion_config["num_train_timesteps"])
    args.beta_schedule = str(world.diffusion_config["beta_schedule"])
    if args.action_dim != len(BUTTONS):
        raise ValueError(f"The demo needs the canonical six-action DiT; config has {args.action_dim}")
    if args.horizon != 1:
        raise ValueError("Public raw demo expects horizon 1")
    bootstrap_src = world.resolve(args.npz_root or "bootstrap/bootstrap.parquet")
    bootstrap_index, bootstrap_obs_full, bootstrap_actions, bootstrap_info = build_bootstrap_segment(
        npz_root=str(bootstrap_src),
        context_frames=int(args.context_frames),
        horizon=int(args.horizon),
        start_index=args.start_index,
        seed=int(args.seed),
    )
    frame_hw = (int(bootstrap_obs_full.shape[-2]), int(bootstrap_obs_full.shape[-1]))
    bootstrap_obs = bootstrap_obs_full[: int(args.context_frames)].contiguous()
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    obs_batched = bootstrap_obs.unsqueeze(0).to(device)
    bootstrap_latents = encode_obs_like_training(vae, obs_batched, sample_latent=bool(args.sample_latent))
    latent_h, latent_w = int(bootstrap_latents.shape[-2]), int(bootstrap_latents.shape[-1])
    if latent_h != int(args.latent_size) or latent_w != int(args.latent_size):
        raise ValueError(
            f"Bootstrap latent size is {(latent_h, latent_w)} but --latent_size is {args.latent_size}. "
            "Check your VAE downsampling, crop size, and DiT config."
        )

    initial_context = bootstrap_latents[0].detach().clone()
    context_queue: Deque[torch.Tensor] = deque(
        [initial_context[i].contiguous() for i in range(int(args.context_frames))],
        maxlen=int(args.context_frames),
    )

    action_history: Deque[List[float]] = deque(maxlen=int(args.context_frames + args.horizon))
    if bool(args.seed_actions_from_bootstrap) and bootstrap_actions is not None:
        for row in bootstrap_actions[: int(args.context_frames + args.horizon)]:
            action_history.append(clamp_action_vector(row.tolist()))

    horizons = torch.full((1,), int(args.horizon), dtype=torch.long, device=device)
    latent_shape = (1, int(args.vae_z_dim), latent_h, latent_w)
    context_noise_bucket = torch.full((1,), int(args.context_noise_bucket), dtype=torch.long, device=device)

    display_w, display_h = compute_display_size(
        frame_hw=frame_hw,
        display_width=int(args.display_width),
        display_height=int(args.display_height),
        display_scale=int(args.display_scale),
    )

    window_flags = cv2.WINDOW_NORMAL
    if hasattr(cv2, "WINDOW_KEEPRATIO"):
        window_flags |= cv2.WINDOW_KEEPRATIO
    cv2.namedWindow(str(args.window_name), window_flags)
    cv2.resizeWindow(str(args.window_name), int(display_w), int(display_h))
    cv2.setMouseCallback(str(args.window_name), mouse_callback)

    writer: Optional[cv2.VideoWriter] = None
    key_times: Dict[int, float] = {}
    hold_s = float(args.key_hold_ms) / 1000.0

    generated_latents: List[np.ndarray] = []
    action_vectors: List[np.ndarray] = []
    step_latencies_s: List[float] = []

    print(f"Bootstrap dataset index: {bootstrap_index}")
    if "path" in bootstrap_info:
        print(f"Bootstrap clip path: {bootstrap_info['path']}")
    print(f"Bootstrap clip length: {bootstrap_info.get('clip_T', 'unknown')}")
    print(f"Device: {device_kind}")
    print("VAE/DiT: strict safetensors loading succeeded")
    print(f"Inference context-noise bucket: {int(args.context_noise_bucket)} / {max(0, int(args.context_noise_levels) - 1)}")
    if str(args.sampler) == "ddim":
        print(f"Sampler: DDIM ({int(args.ddim_steps)} steps, eta={float(args.ddim_eta):.3f})")
    else:
        print(f"Sampler: DDPM ({int(args.diffusion_steps)} steps)")
    print("Controls: WASD move | mouse turn | F/LMB attack | Q/Esc quit")

    last_elapsed = 0.0
    smooth_fps: Optional[float] = None
    step_idx = 0

    while True:
        wait_ms = int(max(1, args.frame_delay_ms))
        key_code = cv2.waitKeyEx(wait_ms)
        now = time.perf_counter()
        norm_code = normalize_key_code(key_code)
        if norm_code not in NO_KEY_CODES:
            update_active_keys(key_times, norm_code, now, hold_s)

        action_vec, should_quit = make_action_from_input(
            active_keys=key_times,
            now=now,
            hold_s=hold_s,
            keyboard_turn_step=float(args.keyboard_turn_step),
            keyboard_look_step=float(args.keyboard_look_step),
            mouse_turn_scale=float(args.mouse_turn_scale),
            mouse_look_scale=float(args.mouse_look_scale),
            invert_mouse_y=bool(args.invert_mouse_y),
        )
        if should_quit:
            break

        action_history.append(action_vec)
        actions, action_mask = action_history_to_tensors(
            history=list(action_history),
            required_len=int(args.context_frames + args.horizon),
            action_dim=int(args.action_dim),
            device=device,
        )

        context_tensor = torch.stack(list(context_queue), dim=0).unsqueeze(0)

        t0 = time.perf_counter()

        def model_fn(xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return model(
                noisy_target=xt,
                context_latents=context_tensor,
                diffusion_t=t,
                horizons=horizons,
                actions=actions,
                action_mask=action_mask,
                context_noise_bucket=context_noise_bucket,
            )

        if str(args.sampler) == "ddim":
            pred_latent = ddim_sample_loop(
                diffusion=diffusion,
                model_fn=model_fn,
                shape=latent_shape,
                device=device,
                num_sample_steps=int(args.ddim_steps),
                eta=float(args.ddim_eta),
            )
        else:
            pred_latent = diffusion.sample_loop(model_fn=model_fn, shape=latent_shape, device=device)
        pred_img = decode_latent_like_training(vae, pred_latent, output_size=frame_hw)
        pred_latent_single = pred_latent[0].detach().clone()
        context_queue.append(pred_latent_single)

        elapsed = time.perf_counter() - t0
        last_elapsed = elapsed
        inst_fps = 1.0 / max(elapsed, 1e-6)
        smooth_fps = inst_fps if smooth_fps is None else (0.9 * smooth_fps + 0.1 * inst_fps)

        generated_latents.append(pred_latent_single.detach().cpu().to(torch.float16).numpy())
        action_vectors.append(np.asarray(action_vec, dtype=np.float32))
        step_latencies_s.append(float(elapsed))

        frame_rgb = tensor_to_uint8_rgb(pred_img)

        target_w, target_h = compute_display_size(
            frame_hw=(frame_rgb.shape[0], frame_rgb.shape[1]),
            display_width=int(args.display_width),
            display_height=int(args.display_height),
            display_scale=int(args.display_scale),
        )
        if frame_rgb.shape[1] != target_w or frame_rgb.shape[0] != target_h:
            interpolation = cv2.INTER_LINEAR if (target_w != frame_rgb.shape[1] or target_h != frame_rgb.shape[0]) else cv2.INTER_NEAREST
            frame_rgb = cv2.resize(frame_rgb, (target_w, target_h), interpolation=interpolation)

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if args.hud == "compact":
            frame_bgr = draw_hud(
                frame_bgr,
                step=step_idx + 1,
                fps=float(smooth_fps),
                action=action_vec,
            )
        cv2.imshow(str(args.window_name), frame_bgr)
        cv2.resizeWindow(str(args.window_name), int(target_w), int(target_h))

        if writer is None and args.record_path:
            writer = ensure_writer(args.record_path, fps=max(1.0, float(smooth_fps)), size_hw=(frame_bgr.shape[0], frame_bgr.shape[1]))
        if writer is not None:
            writer.write(frame_bgr)

        step_idx += 1
        if int(args.max_steps) > 0 and step_idx >= int(args.max_steps):
            break

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    cache_path = derive_latent_cache_path(args.record_path, args.latent_cache_path)
    if cache_path is not None:
        save_rollout_cache(
            out_path=cache_path,
            generated_latents=generated_latents,
            action_vectors=action_vectors,
            step_latencies_s=step_latencies_s,
            bootstrap_obs=bootstrap_obs,
            bootstrap_latents=initial_context,
            bootstrap_info=bootstrap_info,
            start_index=bootstrap_index,
            frame_hw=frame_hw,
            args=args,
        )
        print(f"Saved latent/action cache to: {cache_path}")

    print(f"Finished after {step_idx} generated steps. Last-step latency: {last_elapsed:.3f}s")
    if args.record_path:
        print(f"Saved MP4 to: {args.record_path}")


if __name__ == "__main__":
    main()
