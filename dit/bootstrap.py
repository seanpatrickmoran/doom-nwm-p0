from __future__ import annotations
from pathlib import Path
import pyarrow.parquet as pq
import numpy as np
import torch
import random


_COLUMNS = ("T", "C", "H", "W", "frames", "actions")


def load_bootstrap_segment(
    *,
    source: str,
    context_frames: int,
    horizon: int,
    index: int | None,
    seed: int,
    action_dim: int = 6,
):
    """
    obs:        float32 [context_frames+horizon+1, 3, H, W], in [0, 1]
    actions:    float32 [context_frames+horizon, action_dim]
    """
    path = Path(source).expanduser()
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"Expected a .parquet bootstrap file: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)

    needed = int(context_frames) + int(horizon) + 1
    if int(context_frames) < 1 or int(horizon) < 1 or int(action_dim) < 1:
        raise ValueError("context_frames, horizon and action_dim must be positive")

    table = pq.read_table(path, columns=list(_COLUMNS))
    if not table.num_rows:
        raise ValueError(f"Empty bootstrap dataset: {path}")

    idx = random.Random(int(seed) + 17).randrange(table.num_rows) if index is None else int(index)
    if not 0 <= idx < table.num_rows:
        raise IndexError(f"Bootstrap row {idx} outside [0, {table.num_rows})")

    row = {col: table[col][idx].as_py() for col in _COLUMNS}
    T, C, H, W = (int(row[col]) for col in ("T", "C", "H", "W"))
    if C != 3 or H <= 0 or W <= 0 or T < needed:
        raise ValueError(f"Bootstrap row {idx}: invalid frame shape {(T, C, H, W)} for {needed} frames")

    frame_array = np.frombuffer(row["frames"], dtype=np.uint8)
    if frame_array.size != T * C * H * W:
        raise ValueError(f"Bootstrap row {idx}: malformed frame data")
    observations = torch.from_numpy(frame_array.copy().reshape(T, C, H, W))
    observations = observations[:needed].to(dtype=torch.float32).div_(255.0).contiguous()

    action_array = np.asarray(row["actions"], dtype=np.float32)
    if action_array.size % action_dim:
        raise ValueError(f"Bootstrap row {idx}: action dimensions are not divisible by {action_dim}")
    actions = action_array.reshape(-1, action_dim)
    if len(actions) < needed - 1:
        raise ValueError(f"Bootstrap row {idx}: not enough actions ({len(actions)} for {needed} frames)")
    actions = torch.from_numpy(actions[:needed - 1].copy()).contiguous()

    meta = {"path": path.name, "clip_T": T, "normalized": True}
    return idx, observations, actions, meta
