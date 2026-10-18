from __future__ import annotations

import argparse
import time
from collections import deque

import cv2
import numpy as np
import torch

from dit import infer as base
from dit.hud import action_label, draw_hud
from dit.loading import load_world
from flybywire.loading import load_critic
from flybywire.controller import (FlyByWireRule, TacticalScoreWeights,
                                  OnlineInterventionBudget, LiveLatentFlyByWireController)


def parse_args():
    p = argparse.ArgumentParser(
        "Human -> LiveFlyByWire -> autoregressive video DiT"
    )
    p.add_argument("--model-root", default=None,
                   help="Local HF directory or owner/repo; defaults to MODEL_ROOT")
    p.add_argument("--bootstrap-data", default=None,
                   help="Local Parquet file or repository-relative file")

    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start-index", type=int, default=None)

    # Visible autoregressive generation.
    p.add_argument("--render-ddim-steps", "--ddim_steps",
                   dest="render_ddim_steps", type=int, default=50)
    p.add_argument("--ddim_eta", type=float, default=0.0,
                   help="Visible DDIM eta (steering lookahead remains eta=0).")
    p.add_argument("--sampler", choices=("ddim",), default="ddim",
                   help="Visible sampler; FlyByWire currently supports DDIM only.")
    p.add_argument("--context_noise_bucket", type=int, choices=(0,), default=0,
                   help="Clean-context conditioning; currently only bucket 0 is supported.")
    p.add_argument("--seed_actions_from_bootstrap",
                   action=argparse.BooleanOptionalAction, default=True)

    # Cheap FlyByWire candidate lookahead.
    p.add_argument("--flybywire-ddim-steps", type=int, default=8)

    p.add_argument(
        "--shield-rule",
        default="h32:0.25:0.02:0.025",
    )
    p.add_argument(
        "--score-mode",
        default="value",
        choices=("hazard", "utility", "value"),
    )
    p.add_argument(
        "--candidate-mode",
        default="turn_forward_only",
    )
    p.add_argument(
        "--max-action-distance",
        type=float,
        default=1.05,
    )
    p.add_argument(
        "--target-intervention-rate",
        type=float,
        default=0.05,
    )

    p.add_argument("--mouse-turn-scale", type=float, default=1.0)

    p.add_argument("--display-width", "--display_width",
                   dest="display_width", type=int, default=1280)
    p.add_argument("--display-height", "--display_height",
                   dest="display_height", type=int, default=960)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--hud", choices=("compact", "off"), default="compact")
    p.add_argument("--hud-actions", type=int, default=8,
                   help="Number of recent generated-frame actions in the compact HUD")

    return p.parse_args()


def build_controller(a, device):
    world = load_world(a.model_root, device=device)
    critic, cfg = load_critic(a.model_root, device=device)
    if int(cfg["context_frames"]) != 4:
        raise ValueError("The current FlyByWire interactive demo expects four context frames")
    if int(cfg["context_frames"]) != int(world.dit_config["max_context_frames"]):
        raise ValueError("Critic and DiT context-frame counts disagree")
    if int(cfg["action_dim"]) != 6 or int(world.dit_config["action_dim"]) != 6:
        raise ValueError("FlyByWire demo expects canonical six-action representations")
    if int(cfg["max_action_tokens"]) != int(world.dit_config["max_action_tokens"]):
        raise ValueError("Critic and DiT action-token counts disagree")
    rule = FlyByWireRule.parse(a.shield_rule)
    budget = OnlineInterventionBudget(target_rate=a.target_intervention_rate,
                                      window=512, warmup=64)
    return LiveLatentFlyByWireController(
        world=world, critic=critic, rules=[rule], device=device,
        horizons=tuple(int(h) for h in cfg["horizons"]),
        context_frames=int(cfg["context_frames"]), predict_horizon=2,
        required_action_tokens=int(cfg["max_action_tokens"]),
        turn_scale=10.0, ddim_steps=a.flybywire_ddim_steps,
        shield_every=1, preserve_attack=True,
        candidate_mode=a.candidate_mode,
        emergency_candidate_mode=a.candidate_mode,
        emergency_risk_threshold=2.0,
        max_action_distance=a.max_action_distance,
        score_mode=a.score_mode,
        score_weights=TacticalScoreWeights(), online_budget=budget,
        common_ddim_noise=True,
    )


@torch.no_grad()
def main():
    a = parse_args()

    base.set_seed(a.seed)

    device, _ = base.detect_device(a.device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    controller = build_controller(a, device)
    world = controller.world
    idx, obs_full, bootstrap_actions, info = \
        base.build_bootstrap_segment(
            npz_root=str(world.resolve(a.bootstrap_data or "bootstrap/bootstrap.parquet")),
            context_frames=4,
            horizon=1,
            start_index=a.start_index,
            seed=a.seed,
        )

    obs = obs_full[:4].contiguous()
    frame_hw = (
        int(obs.shape[-2]),
        int(obs.shape[-1]),
    )

    z = world.encode_obs(
        obs.unsqueeze(0).to(device)
    )[0]

    latent_ctx = deque(
        [
            z[i].detach()
            for i in range(4)
        ],
        maxlen=4,
    )

    image_ctx = deque(
        [obs[i].cpu().numpy().astype(np.float32) for i in range(4)],
        maxlen=4,
    )

    action_hist = deque(maxlen=5)
    if a.hud_actions < 1:
        raise ValueError("--hud-actions must be >= 1")
    action_ticker = deque(maxlen=a.hud_actions)

    if a.seed_actions_from_bootstrap and bootstrap_actions is not None:
        for row in bootstrap_actions[:5]:
            action_hist.append(
                np.asarray(
                    base.clamp_action_vector(row.tolist()),
                    dtype=np.float32,
                )
            )

            #OCV window
    name = "DiT + LiveFlyByWire"
    flags = (cv2.WINDOW_NORMAL| getattr(cv2, "WINDOW_KEEPRATIO", 0))
    cv2.namedWindow(name, flags)
    cv2.resizeWindow(name, a.display_width, a.display_height,)
    cv2.setMouseCallback(name, base.mouse_callback,)

    keys = {}
    hold_s = 0.140

    enabled = True
    c_latch = False
    step = 0
    smooth_fps = None

    print(
        f"bootstrap={idx} "
        f"source={info.get('path', 'unknown')}"
    )
    print(
        f"FlyByWire: "
        f"h2/DDIM-{a.flybywire_ddim_steps}, "
        f"{a.score_mode}, "
        f"{a.candidate_mode}"
    )
    print(
        f"visible generation: "
        f"h1/DDIM-{a.render_ddim_steps}"
    )
    print(
        "WASD | mouse turn | LMB/F attack | "
        "C toggle FlyByWire | Q/Esc quit"
    )

    try:
        while True:
            key = cv2.waitKeyEx(1)
            now = time.perf_counter()

            key = base.normalize_key_code(key)

            if key not in base.NO_KEY_CODES:
                base.update_active_keys(
                    keys,
                    key,
                    now,
                    hold_s,
                )

            # Runtime FlyByWire toggle.
            c = base.key_is_active(
                keys,
                now,
                hold_s,
                [ord("c"), ord("C")],
            )

            if c and not c_latch:
                enabled = not enabled
                c_latch = True
                print(
                    f"[FlyByWire] "
                    f"{'ON' if enabled else 'OFF'}"
                )
            elif not c:
                c_latch = False

            user, quit_now = \
                base.make_action_from_input(
                    keys,
                    now,
                    hold_s,
                    keyboard_turn_step=6.0,
                    keyboard_look_step=4.0,
                    mouse_turn_scale=a.mouse_turn_scale,
                    mouse_look_scale=0.0,
                    invert_mouse_y=False,
                )

            if quit_now:
                break

            user = np.asarray(
                user,
                dtype=np.float32,
            )

            user[5] = np.clip(
                user[5],
                -10.0,
                10.0,
            )

            #Live capture
            decision = None
            executed = user.copy()

            t0 = time.perf_counter()

            #dont intervene if NOOP.
            if (
                enabled
                and not np.allclose(
                    user,
                    0.0,
                    atol=1e-6,
                )
            ):
                decision = controller.choose(
                    np.stack(image_ctx),
                    list(action_hist),
                    user,
                )

                executed = np.asarray(
                    decision.action,
                    dtype=np.float32,
                ).reshape(6)

            control_ms = (
                1000.0
                * (time.perf_counter() - t0)
            )

            action_hist.append(
                executed.copy()
            )

            actions, mask = \
                base.action_history_to_tensors(
                    list(action_hist),
                    required_len=5,
                    action_dim=6,
                    device=device,
                )

            context = torch.stack(
                list(latent_ctx)
            ).unsqueeze(0)

            t0 = time.perf_counter()

            pred_z = world.predict_latent(
                context,
                actions,
                horizon=1,
                action_mask=mask,

                # Clean inference conditioning.
                context_noise_bucket=torch.full(
                    (1,), int(a.context_noise_bucket),
                    dtype=torch.long, device=device,
                ),

                sampler=a.sampler,
                ddim_steps=a.render_ddim_steps,
                eta=a.ddim_eta,
            )

            pred_img = world.decode_latent(
                pred_z,
                output_size=frame_hw,
            )

            render_ms = (
                1000.0
                * (time.perf_counter() - t0)
            )

            latent_ctx.append(
                pred_z[0].detach()
            )

            image_ctx.append(
                pred_img[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            #render here
            frame = base.tensor_to_uint8_rgb(
                pred_img
            )

            frame = cv2.resize(
                frame,
                (a.display_width, a.display_height),
                interpolation=cv2.INTER_LINEAR,
            )
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            action_ticker.append((
                action_label(executed),
                bool(decision is not None and decision.intervened),
            ))
            # FPS includes FlyByWire candidate scoring plus visible generation.
            elapsed_s = max((control_ms + render_ms) / 1000.0, 1e-6)
            inst_fps = 1.0 / elapsed_s
            smooth_fps = (inst_fps if smooth_fps is None
                          else 0.9 * smooth_fps + 0.1 * inst_fps)
            if a.hud == "compact":
                frame = draw_hud(
                    frame,
                    step=step + 1,
                    fps=float(smooth_fps),
                    action=executed,
                    fbw_enabled=enabled,
                    recent_actions=action_ticker,
                )

            cv2.imshow(name, frame)

            step += 1

            if (
                a.max_steps > 0
                and step >= a.max_steps
            ):
                break

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
