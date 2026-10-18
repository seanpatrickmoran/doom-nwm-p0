from __future__ import annotations
from typing import Sequence
import numpy as np
DEFAULT_TURN_SCALE = 10.0

def action_distance(
    a: np.ndarray,
    b: np.ndarray,
    *,
    turn_scale: float = DEFAULT_TURN_SCALE,
) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)

    button = np.abs(aa[..., :5] - bb[..., :5]).sum(axis=-1)
    turn = np.abs(aa[..., 5] - bb[..., 5]) / max(1e-6, float(turn_scale))
    return button + turn


def local_action_candidates(
    base_action: Sequence[float] | np.ndarray,
    *,
    turn_scale: float = DEFAULT_TURN_SCALE,
    turn_steps: Sequence[float] = (-1.0, -0.5, 0.0, 0.5, 1.0),
    include_discrete_edits: bool = True,
    preserve_attack: bool = False,
    candidate_mode: str = "legacy",
    max_distance: float | None = None,
    turn_delta_steps: Sequence[float] = (-0.25, -0.125, 0.125, 0.25),
) -> np.ndarray:
    base = np.asarray(base_action, dtype=np.float32).reshape(6).copy()
    base[:5] = (base[:5] > 0.5).astype(np.float32)
    base[5] = float(np.clip(base[5], -turn_scale, turn_scale))

    mode = str(candidate_mode or "legacy").strip().lower().replace("-", "_")

    if mode in {"old", "broad"}:
        mode = "legacy"
    if mode in {"turn", "turn_only"}:
        mode = "micro_turn"
    if mode in {"turn_forward", "forward_turn", "turn_forward_only", "turn_and_forward"}:
        mode = "turn_forward_only"
    if mode in {"small", "residual_small"}:
        mode = "residual"
    if mode in {"intent", "intent_evasive", "intent_preserving", "evasive", "evasive_intent"}:
        mode = "intent_evasive"
    if mode in {
        "intent_keep_lateral",
        "intent_evasive_keep_lateral",
        "intent_preserving_keep_lateral",
        "evasive_keep_lateral",
        "keep_lateral",
    }:
        mode = "intent_evasive_keep_lateral"

    candidates: list[np.ndarray] = [base.copy()]

    def add(cand: np.ndarray) -> None:
        c = np.asarray(cand, dtype=np.float32).reshape(6).copy()
        c[:5] = (c[:5] > 0.5).astype(np.float32)
        c[5] = float(np.clip(c[5], -turn_scale, turn_scale))

        if preserve_attack:
            c[4] = base[4]

        candidates.append(c)

    if mode == "legacy":
        for step in turn_steps:
            cand = base.copy()
            cand[5] = float(
                np.clip(float(step) * float(turn_scale), -turn_scale, turn_scale)
            )
            add(cand)

        if include_discrete_edits:
            # Suppress forward/backward separately.
            cand = base.copy()
            cand[0] = 0.0
            cand[1] = 0.0
            add(cand)

            # No-op movement but keep turn/attack.
            cand = base.copy()
            cand[0:4] = 0.0
            add(cand)

            # Strafe alternatives.
            for lr in (0, 1, 2):
                cand = base.copy()
                cand[2] = 1.0 if lr == 1 else 0.0
                cand[3] = 1.0 if lr == 2 else 0.0
                add(cand)

            # Attack toggle.
            if not preserve_attack:
                cand = base.copy()
                cand[4] = 1.0 - cand[4]
                add(cand)
            else:
                cand = base.copy()
                cand[4] = base[4]
                add(cand)

    elif mode in {"micro_turn", "turn_forward_only", "intent_evasive", "intent_evasive_keep_lateral", "residual", "movement_small", "emergency"}:
        for delta in turn_delta_steps:
            cand = base.copy()
            cand[5] = float(base[5] + float(delta) * float(turn_scale))
            add(cand)

        for factor in (0.5, 0.0):
            cand = base.copy()
            cand[5] = float(base[5] * float(factor))
            add(cand)

        if include_discrete_edits and mode == "turn_forward_only":
            if base[0] > 0.5:
                cand = base.copy()
                cand[0] = 0.0
                cand[4] = base[4]
                add(cand)

        if include_discrete_edits and mode == "intent_evasive":
            if base[2] <= 0.5 and base[3] <= 0.5:
                cand = base.copy()
                cand[2] = 1.0
                add(cand)

                cand = base.copy()
                cand[3] = 1.0
                add(cand)
            else:
                cand = base.copy()
                cand[2] = 0.0
                cand[3] = 0.0
                add(cand)

                cand = base.copy()
                cand[2] = 1.0 - base[2]
                cand[3] = 1.0 - base[3]
                add(cand)

        if include_discrete_edits and mode == "intent_evasive_keep_lateral":
            if base[2] <= 0.5 and base[3] <= 0.5:
                cand = base.copy()
                cand[2] = 1.0
                add(cand)

                cand = base.copy()
                cand[3] = 1.0
                add(cand)

        if include_discrete_edits and mode in {"residual", "movement_small", "emergency"}:
            for idx in (0, 1, 2, 3):
                if base[idx] > 0.5:
                    cand = base.copy()
                    cand[idx] = 0.0
                    add(cand)

            if base[2] <= 0.5 and base[3] <= 0.5:
                cand = base.copy()
                cand[2] = 1.0
                add(cand)

                cand = base.copy()
                cand[3] = 1.0
                add(cand)

        if include_discrete_edits and mode in {"movement_small", "emergency"}:
            for lr in (0, 1, 2):
                cand = base.copy()
                cand[2] = 1.0 if lr == 1 else 0.0
                cand[3] = 1.0 if lr == 2 else 0.0
                add(cand)

        if include_discrete_edits and mode == "emergency":
            cand = base.copy()
            cand[0] = 0.0
            cand[1] = 0.0
            add(cand)

            cand = base.copy()
            cand[0:4] = 0.0
            add(cand)

            if not preserve_attack:
                cand = base.copy()
                cand[4] = 1.0 - cand[4]
                add(cand)

    else:
        raise ValueError(
            f"Unknown candidate_mode={candidate_mode!r}; expected legacy, micro_turn, "
            "intent_evasive, intent_evasive_keep_lateral, turn_forward_only, "
            "residual, movement_small, or emergency."
        )

    seen: set[tuple[float, ...]] = set()
    out: list[np.ndarray] = []

    for cand in candidates:
        cand[:5] = (cand[:5] > 0.5).astype(np.float32)
        cand[5] = float(np.clip(cand[5], -turn_scale, turn_scale))

        if preserve_attack:
            cand[4] = base[4]

        if max_distance is not None:
            d = float(action_distance(cand[None, :], base[None, :], turn_scale=turn_scale)[0])
            if d > float(max_distance) and not np.allclose(cand, base, atol=1e-6):
                continue

        key = tuple(float(x) for x in cand.tolist())
        if key not in seen:
            out.append(cand.astype(np.float32, copy=True))
            seen.add(key)

    if not out:
        out = [base.astype(np.float32, copy=True)]

    return np.stack(out, axis=0)


