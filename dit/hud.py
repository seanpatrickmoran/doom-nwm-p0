from __future__ import annotations

from typing import Iterable, Optional, Sequence

import cv2
import numpy as np


def action_label(action: Sequence[float]) -> str:
    """Format [forward, backward, left, right, attack, turn] action."""
    if len(action) != 6:
        raise ValueError(f"Expected a six-dimensional action, received {len(action)}")
    labels = [short for idx, short in enumerate(("F", "B", "L", "R", "A"))
              if float(action[idx]) > 0.5]
    turn = float(action[5])
    if abs(turn) >= 0.05:
        labels.append(f"T{turn:+.1f}")
    return "+".join(labels) if labels else "NOOP"


def draw_hud(
    frame_bgr: np.ndarray,
    *,
    step: int,
    fps: float,
    action: Sequence[float],
    fbw_enabled: Optional[bool] = None,
    recent_actions: Iterable[tuple[str, bool]] = (),
) -> np.ndarray:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("draw_hud expects a BGR frame of shape [H,W,3]")
    frame = frame_bgr.copy()
    height, width = frame.shape[:2]
    if width < 180 or height < 48:
        return frame

    font = cv2.FONT_HERSHEY_SIMPLEX
    size = max(0.38, min(0.62, width / 1280.0 * 0.58))
    thickness = 1
    white = (238, 238, 238)  # OpenCV BGR
    blue = (255, 150, 55)
    muted = (165, 165, 165)
    pad = 12
    row1_y = 29
    row2_y = 61
    bar_height = 43 if fbw_enabled is None else 76
    bar_height = min(bar_height, height)

    frame[:bar_height] = (frame[:bar_height].astype(np.float32) * 0.35).astype(np.uint8)
    line = f"STEP {step:05d}  |  FPS {fps:5.1f}  |  ACTION {action_label(action)}"
    cv2.putText(frame, line, (pad, row1_y), font, size, white,
                thickness, cv2.LINE_AA)

    if fbw_enabled is None or height < 85:
        return frame

    status = f"FBW {'ON' if fbw_enabled else 'OFF'}"
    cv2.putText(frame, status, (pad, row2_y), font, size, white,
                thickness, cv2.LINE_AA)
    if not fbw_enabled:
        return frame

    prefix = status + "  |"
    prefix_width = cv2.getTextSize(prefix, font, size, thickness)[0][0]
    cursor = pad + prefix_width + 12
    cv2.putText(frame, "|", (pad + cv2.getTextSize(status, font, size, thickness)[0][0] + 9,
                            row2_y), font, size, muted, thickness, cv2.LINE_AA)

    available = max(0, width - pad - cursor)
    shown: list[tuple[str, bool, int]] = []
    for label, intervened in reversed(list(recent_actions)):
        text = ("*" if intervened else "") + label
        text_width = cv2.getTextSize(text, font, size, thickness)[0][0] + 20
        if text_width > available:
            break
        shown.append((text, intervened, text_width))
        available -= text_width

    items = list(reversed(shown))
    for i, (text, intervened, text_width) in enumerate(items):
        cv2.putText(frame, text, (cursor, row2_y), font, size,
                    blue if intervened else white, thickness, cv2.LINE_AA)
        cursor += text_width
        if i < len(items) - 1:
            cv2.putText(frame, "|", (cursor - 11, row2_y), font, size,
                        muted, thickness, cv2.LINE_AA)
    return frame
