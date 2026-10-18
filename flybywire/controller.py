from __future__ import annotations
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np
import torch
from .actions import DEFAULT_TURN_SCALE, action_distance, local_action_candidates
from .critic import LatentActionHazardCritic

@dataclass(frozen=True)
class FlyByWireRule:
    horizon: int
    threshold: float
    margin: float
    distance_penalty: float
    name: str = ""

    @classmethod
    def parse(cls, text: str) -> "FlyByWireRule":
        raw = str(text).strip()
        if not raw:
            raise ValueError("empty shield rule")
        parts = raw.replace("/", ":").split(":")
        if len(parts) < 4:
            raise ValueError(
                f"Shield rule {text!r} must look like h32:0.20:0.01:0.01 "
                "meaning horizon:threshold:margin:distance_penalty"
            )
        htxt = parts[0].strip().lower()
        if htxt.startswith("h"):
            htxt = htxt[1:]
        horizon = int(htxt)
        threshold = float(parts[1])
        margin = float(parts[2])
        distance_penalty = float(parts[3])
        name = parts[4] if len(parts) > 4 else f"h{horizon}_t{threshold:g}_m{margin:g}_d{distance_penalty:g}"
        return cls(horizon=horizon, threshold=threshold, margin=margin, distance_penalty=distance_penalty, name=name)


@dataclass(frozen=True)
class TacticalScoreWeights:
    death: float = 1.0
    health_loss: float = 0.02
    damage_taken: float = 0.01
    frag_gain: float = 1.0
    damage_dealt: float = 0.02
    reward: float = 0.0


class OnlineInterventionBudget:
    def __init__(
        self,
        *,
        target_rate: float = 0.08,
        window: int = 512,
        warmup: int = 64,
        min_margin: float = 0.0,
        max_margin: float = 1e9,
    ) -> None:
        self.target_rate = float(np.clip(target_rate, 1e-4, 1.0))
        self.window = max(8, int(window))
        self.warmup = max(0, int(warmup))
        self.min_margin = float(min_margin)
        self.max_margin = float(max_margin)
        self.improvements: deque[float] = deque(maxlen=self.window)
        self.interventions: deque[float] = deque(maxlen=self.window)

    def effective_margin(self, fallback: float) -> float:
        base = max(float(fallback), self.min_margin)
        if len(self.improvements) < self.warmup:
            return base
        vals = np.asarray([x for x in self.improvements if math.isfinite(float(x))], dtype=np.float32)
        if vals.size == 0:
            return base
        q = float(np.clip(1.0 - self.target_rate, 0.0, 1.0))
        dyn = float(np.quantile(vals, q))
        return float(np.clip(max(base, dyn), self.min_margin, self.max_margin))

    def update(self, *, improvement: float, intervened: bool) -> None:
        if math.isfinite(float(improvement)):
            self.improvements.append(float(improvement))
        self.interventions.append(1.0 if bool(intervened) else 0.0)

    @property
    def intervention_rate(self) -> float:
        if not self.interventions:
            return 0.0
        return float(np.mean(np.asarray(self.interventions, dtype=np.float32)))


@dataclass(slots=True)
class LiveDecision:
    action: np.ndarray
    base_action: np.ndarray
    candidate_action: np.ndarray
    intervened: bool
    rule_name: str
    horizon: int
    base_risk: float
    chosen_risk: float
    chosen_index: int
    improvement: float
    distance: float
    threshold: float
    margin: float
    distance_penalty: float
    risk_by_horizon: dict[int, float]
    score_mode: str
    candidate_mode: str
    base_score: float
    chosen_score: float
    budget_margin: float
    budget_rate: float
    base_terms: dict[str, float]
    chosen_terms: dict[str, float]


def _pad_action_history(history: Sequence[Sequence[float]] | np.ndarray, candidate: Sequence[float] | np.ndarray, required_len: int) -> np.ndarray:
    hist = np.asarray(history, dtype=np.float32)
    if hist.size == 0:
        hist = np.zeros((0, 6), dtype=np.float32)
    if hist.ndim == 1:
        hist = hist.reshape(1, -1)
    if hist.shape[-1] != 6:
        raise ValueError(f"Expected action history [...,6], got {hist.shape}")
    cand = np.asarray(candidate, dtype=np.float32).reshape(1, 6)
    seq = np.concatenate([hist, cand], axis=0)
    if seq.shape[0] < int(required_len):
        pad = np.zeros((int(required_len) - seq.shape[0], 6), dtype=np.float32)
        seq = np.concatenate([pad, seq], axis=0)
    elif seq.shape[0] > int(required_len):
        seq = seq[-int(required_len):]
    return seq.astype(np.float32)


class LiveLatentFlyByWireController:
    def __init__(
        self,
        *,
        world: Any,
        critic: LatentActionHazardCritic,
        rules: Sequence[FlyByWireRule],
        device: torch.device,
        horizons: Sequence[int],
        context_frames: int = 4,
        predict_horizon: int = 2,
        required_action_tokens: int = 6,
        turn_scale: float = 10.0,
        ddim_steps: int = 8,
        shield_every: int = 1,
        preserve_attack: bool = True,
        candidate_mode: str = "residual",
        emergency_candidate_mode: str = "emergency",
        emergency_horizon: int = 4,
        emergency_risk_threshold: float = 0.80,
        max_action_distance: float = 1.25,
        lateral_suppression_min_improvement: float = 0.0,
        lateral_suppression_penalty: float = 0.0,
        lateral_flip_penalty: float = 0.0,
        forward_suppression_penalty: float = 0.0,
        score_mode: str = "utility",
        score_weights: TacticalScoreWeights | None = None,
        online_budget: OnlineInterventionBudget | None = None,
        common_ddim_noise: bool = True,
    ) -> None:
        self.world = world.to(device).eval()
        self.critic = critic.to(device).eval()
        self.rules = list(rules)
        if not self.rules:
            raise ValueError("At least one fly-by-wire rule is required")
        self.device = torch.device(device)
        self.horizons = tuple(int(x) for x in horizons)
        self.horizon_to_idx = {int(h): i for i, h in enumerate(self.horizons)}
        for rule in self.rules:
            if int(rule.horizon) not in self.horizon_to_idx:
                raise ValueError(f"Rule horizon {rule.horizon} not in critic horizons {self.horizons}")
        self.context_frames = int(context_frames)
        self.predict_horizon = int(predict_horizon)
        self.required_action_tokens = int(required_action_tokens)
        self.turn_scale = float(turn_scale)
        self.ddim_steps = int(ddim_steps)
        self.shield_every = max(1, int(shield_every))
        self.preserve_attack = bool(preserve_attack)
        self.candidate_mode = str(candidate_mode)
        self.emergency_candidate_mode = str(emergency_candidate_mode)
        self.emergency_horizon = int(emergency_horizon)
        self.emergency_risk_threshold = float(emergency_risk_threshold)
        self.max_action_distance = float(max_action_distance)
        self.lateral_suppression_min_improvement = max(0.0, float(lateral_suppression_min_improvement))
        self.lateral_suppression_penalty = max(0.0, float(lateral_suppression_penalty))
        self.lateral_flip_penalty = max(0.0, float(lateral_flip_penalty))
        self.forward_suppression_penalty = max(0.0, float(forward_suppression_penalty))
        self.score_mode = str(score_mode).strip().lower()
        if self.score_mode not in {"hazard", "utility", "value", "advantage"}:
            raise ValueError(
                f"score_mode must be 'hazard', 'utility', 'value', or 'advantage', got {score_mode!r}"
            )
        self.score_weights = score_weights or TacticalScoreWeights()
        self.online_budget = online_budget
        self.common_ddim_noise = bool(common_ddim_noise)
        self._calls = 0
        if bool(getattr(self.critic, "use_residual_latent", False)):
            # Live inference has no true target latent, so residual is zero-filled inside the critic.
            # This is acceptable for a smoke/live prototype but thresholds from teacher_residual sweeps
            # may not transfer perfectly. A DDIM-trained critic is the cleaner live model.
            print(
                "[warning] critic was trained/configured with residual_latent features. "
                "Live execution has no teacher target, so residual features are zero-filled. "
                "Consider training a feature_mode=ddim_prediction critic for fully matched live control.",
                flush=True,
            )

    @torch.no_grad()
    def _score_candidates(self, world_context: np.ndarray, action_history: Sequence[Sequence[float]], candidates: np.ndarray) -> dict[str, torch.Tensor]:
        obs = torch.as_tensor(world_context, device=self.device)
        if obs.ndim != 4:
            raise ValueError(f"Expected world context [T,C,H,W], got {tuple(obs.shape)}")
        if int(obs.shape[0]) != self.context_frames:
            raise ValueError(f"Expected {self.context_frames} context frames, got {obs.shape[0]}")
        context_latents_single = self.world.encode_obs(obs.unsqueeze(0))[0]
        seqs = np.stack([
            _pad_action_history(action_history, cand, self.required_action_tokens)
            for cand in np.asarray(candidates, dtype=np.float32)
        ], axis=0)
        n = int(seqs.shape[0])
        context_latents = context_latents_single.unsqueeze(0).expand(n, -1, -1, -1, -1).contiguous()
        action_seq = torch.as_tensor(seqs, device=self.device, dtype=torch.float32)
        action_mask = torch.ones(action_seq.shape[:2], device=self.device, dtype=torch.bool)
        init_noise = None
        if self.common_ddim_noise:
            # Common random numbers make candidate comparisons much less noisy:
            # every action candidate starts from the same DDIM noise sample.
            _tc, z, h, w = context_latents_single.shape
            one = torch.randn((1, z, h, w), device=self.device, dtype=torch.float32)
            init_noise = one.expand(n, -1, -1, -1).contiguous()
        predicted = self.world.predict_latent(
            context_latents,
            action_seq,
            horizon=int(self.predict_horizon),
            action_mask=action_mask,
            sampler="ddim",
            ddim_steps=int(self.ddim_steps),
            init_noise=init_noise,
        )
        out = self.critic(
            context_latents,
            action_seq,
            action_mask=action_mask,
            predicted_latent=predicted,
            residual_latent=None,
        )
        return {k: v.detach().cpu() for k, v in out.items()}

    def _candidate_scores(
        self,
        out: dict[str, torch.Tensor],
        horizon_idx: int,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        hidx = int(horizon_idx)

        def _at(
            name: str,
            *,
            fallback: np.ndarray | None = None,
            required: bool = False,
        ) -> np.ndarray:
            tensor = out.get(name)
            if tensor is None:
                if required:
                    available = ", ".join(sorted(str(k) for k in out.keys()))
                    raise KeyError(
                        f"score_mode={self.score_mode!r} requires critic output {name!r}, "
                        f"but it was not present. Available outputs: {available}. "
                        "Make sure models.py has tactical_value_head, the checkpoint was "
                        "trained with --tactical-value-weight, and live_flybywire.py is "
                        "loading that new checkpoint."
                    )
                if fallback is None:
                    raise KeyError(f"Missing critic output {name!r} and no fallback was provided")
                return fallback.astype(np.float32, copy=False)

            return tensor[..., hidx].detach().cpu().numpy().astype(np.float32)

        death = (
            torch.sigmoid(out["death_logits"])[..., hidx]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        health = _at("health_loss")

        damage_tensor = out.get("damage_taken", out.get("damage"))
        if damage_tensor is None:
            damage_taken = np.zeros_like(death, dtype=np.float32)
        else:
            damage_taken = (
                damage_tensor[..., hidx]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        zeros = np.zeros_like(death, dtype=np.float32)

        frag = _at("frag_gain", fallback=zeros)
        dealt = _at("damage_dealt", fallback=zeros)
        rew = _at("reward", fallback=zeros)

        tactical_value = _at(
            "tactical_value",
            fallback=zeros,
            required=(self.score_mode == "value"),
        )
        tactical_advantage = _at(
            "tactical_advantage",
            fallback=zeros,
            required=(self.score_mode == "advantage"),
        )

        if self.score_mode == "hazard":
            score = death.astype(np.float32, copy=False)

        elif self.score_mode == "value":
            # Higher tactical_value is better; the controller minimizes score.
            score = (-tactical_value).astype(np.float32, copy=False)

        elif self.score_mode == "advantage":
            # Higher centered advantage is better; the controller minimizes score.
            score = (-tactical_advantage).astype(np.float32, copy=False)

        elif self.score_mode == "utility":
            w = self.score_weights
            score = (
                float(w.death) * death
                + float(w.health_loss) * health
                + float(w.damage_taken) * damage_taken
                - float(w.frag_gain) * frag
                - float(w.damage_dealt) * dealt
                - float(w.reward) * rew
            ).astype(np.float32)

        else:
            raise ValueError(f"Unknown score_mode={self.score_mode!r}")

        terms = {
            "death_risk": death,
            "health_loss": health,
            "damage_taken": damage_taken,
            "frag_gain": frag,
            "damage_dealt": dealt,
            "reward": rew,
            "tactical_value": tactical_value,
            "tactical_advantage": tactical_advantage,
            "score": score,
        }
        return score, terms

    def _candidate_edit_penalties(self, base: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        base = np.asarray(base, dtype=np.float32).reshape(6)
        cands = np.asarray(candidates, dtype=np.float32).reshape(-1, 6)
        penalties = np.zeros((cands.shape[0],), dtype=np.float32)

        if self.lateral_suppression_penalty > 0.0:
            suppresses_lateral = (
                ((base[2] > 0.5) & (cands[:, 2] <= 0.5))
                | ((base[3] > 0.5) & (cands[:, 3] <= 0.5))
            )
            penalties += suppresses_lateral.astype(np.float32) * float(self.lateral_suppression_penalty)

        if self.lateral_flip_penalty > 0.0:
            flips_lateral = (
                ((base[2] > 0.5) & (cands[:, 3] > 0.5))
                | ((base[3] > 0.5) & (cands[:, 2] > 0.5))
            )
            penalties += flips_lateral.astype(np.float32) * float(self.lateral_flip_penalty)

        if self.forward_suppression_penalty > 0.0:
            suppresses_forward = (base[0] > 0.5) & (cands[:, 0] <= 0.5)
            penalties += suppresses_forward.astype(np.float32) * float(self.forward_suppression_penalty)

        penalties[0] = 0.0
        return penalties

    @staticmethod
    def _terms_at(terms: dict[str, np.ndarray], idx: int) -> dict[str, float]:
        return {k: float(np.asarray(v).reshape(-1)[int(idx)]) for k, v in terms.items()}

    def choose(self, world_context: np.ndarray, action_history: Sequence[Sequence[float]], base_action: np.ndarray) -> LiveDecision:
        self._calls += 1
        base = np.asarray(base_action, dtype=np.float32).reshape(6).copy()
        empty_terms: dict[str, float] = {}
        if (self._calls - 1) % self.shield_every != 0:
            return LiveDecision(
                action=base,
                base_action=base,
                candidate_action=base,
                intervened=False,
                rule_name="skipped",
                horizon=-1,
                base_risk=float("nan"),
                chosen_risk=float("nan"),
                chosen_index=0,
                improvement=0.0,
                distance=0.0,
                threshold=float("nan"),
                margin=float("nan"),
                distance_penalty=float("nan"),
                risk_by_horizon={},
                score_mode=self.score_mode,
                candidate_mode=self.candidate_mode,
                base_score=float("nan"),
                chosen_score=float("nan"),
                budget_margin=float("nan"),
                budget_rate=float("nan"),
                base_terms=empty_terms,
                chosen_terms=empty_terms,
            )

        mode_used = self.candidate_mode
        candidates = local_action_candidates(
            base,
            turn_scale=self.turn_scale,
            preserve_attack=self.preserve_attack,
            candidate_mode=mode_used,
            max_distance=self.max_action_distance,
        )
        out = self._score_candidates(world_context, action_history, candidates)
        risks = torch.sigmoid(out["death_logits"]).numpy().astype(np.float32)  # [N,H]
        base_risk_by_h = {int(h): float(risks[0, i]) for i, h in enumerate(self.horizons)}

        if (
            self.emergency_risk_threshold > 0.0
            and self.emergency_horizon in self.horizon_to_idx
            and float(base_risk_by_h.get(self.emergency_horizon, 0.0)) >= float(self.emergency_risk_threshold)
            and str(self.emergency_candidate_mode) != str(self.candidate_mode)
        ):
            mode_used = self.emergency_candidate_mode
            candidates = local_action_candidates(
                base,
                turn_scale=self.turn_scale,
                preserve_attack=self.preserve_attack,
                candidate_mode=mode_used,
                max_distance=max(self.max_action_distance, 2.0),
            )
            out = self._score_candidates(world_context, action_history, candidates)
            risks = torch.sigmoid(out["death_logits"]).numpy().astype(np.float32)
            base_risk_by_h = {int(h): float(risks[0, i]) for i, h in enumerate(self.horizons)}

        distances = action_distance(candidates, base[None, :], turn_scale=self.turn_scale).astype(np.float32)
        best_decision: LiveDecision | None = None
        best_rule_score = -float("inf")
        for rule in self.rules:
            idx = self.horizon_to_idx[int(rule.horizon)]
            score, terms = self._candidate_scores(out, idx)
            edit_penalty = self._candidate_edit_penalties(base, candidates)
            penalized = score + float(rule.distance_penalty) * distances + edit_penalty
            terms = dict(terms)
            terms["edit_penalty"] = edit_penalty
            terms["penalized_score"] = penalized
            if self.lateral_suppression_min_improvement > 0.0:
                suppresses_lateral = (
                    ((base[2] > 0.5) & (candidates[:, 2] <= 0.5))
                    | ((base[3] > 0.5) & (candidates[:, 3] <= 0.5))
                )
                penalized_improvement = float(penalized[0]) - penalized
                blocked = suppresses_lateral & (penalized_improvement < self.lateral_suppression_min_improvement)
                if bool(np.any(blocked)):
                    penalized = penalized.copy()
                    penalized[blocked] = np.inf
            best_idx = int(np.argmin(penalized))
            base_risk = float(risks[0, idx])
            chosen_risk = float(risks[best_idx, idx])
            base_score = float(score[0])
            chosen_score = float(score[best_idx])
            improve = base_score - chosen_score
            dist = float(distances[best_idx])
            budget_margin = float(rule.margin)
            if self.online_budget is not None:
                budget_margin = self.online_budget.effective_margin(float(rule.margin))
            intervened = bool(
                base_risk > float(rule.threshold)
                and best_idx != 0
                and improve > float(budget_margin)
                and dist <= float(self.max_action_distance)
            )
            if self.online_budget is not None:
                self.online_budget.update(improvement=improve, intervened=intervened)
            budget_rate = float("nan") if self.online_budget is None else self.online_budget.intervention_rate
            # Prefer an active rule with larger margin-normalized utility gain.
            rule_score = (improve - float(budget_margin)) / max(1e-6, abs(float(rule.threshold))) if intervened else -float("inf")
            candidate_decision = LiveDecision(
                action=candidates[best_idx].astype(np.float32) if intervened else base,
                base_action=base.astype(np.float32),
                candidate_action=candidates[best_idx].astype(np.float32),
                intervened=intervened,
                rule_name=rule.name or f"h{rule.horizon}",
                horizon=int(rule.horizon),
                base_risk=base_risk,
                chosen_risk=chosen_risk,
                chosen_index=best_idx,
                improvement=float(improve),
                distance=dist,
                threshold=float(rule.threshold),
                margin=float(rule.margin),
                distance_penalty=float(rule.distance_penalty),
                risk_by_horizon=dict(base_risk_by_h),
                score_mode=self.score_mode,
                candidate_mode=mode_used,
                base_score=base_score,
                chosen_score=chosen_score,
                budget_margin=budget_margin,
                budget_rate=budget_rate,
                base_terms=self._terms_at(terms, 0),
                chosen_terms=self._terms_at(terms, best_idx),
            )
            if intervened and rule_score > best_rule_score:
                best_rule_score = rule_score
                best_decision = candidate_decision
            elif best_decision is None:
                best_decision = candidate_decision
        assert best_decision is not None
        return best_decision


