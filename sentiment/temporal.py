"""Temporal smoothing of the per-frame emotion estimate via a Bayesian filter.

Per-frame predictions are jittery. Rather than a naive moving average, we run a
small Hidden-Markov-style forward filter: a "sticky" transition matrix encodes
the prior that emotions persist for a while, and each frame's estimate is folded
in as an observation likelihood. This is a classic probabilistic-graphical-model
technique — a nice contrast to black-box temporal nets, and it visibly steadies
the output in the demo.

    belief_t  ∝  (Tᵀ · belief_{t-1})  ⊙  observation_t
"""
from __future__ import annotations

import numpy as np


class EmotionFilter:
    def __init__(self, states: list[str], stickiness: float = 0.85) -> None:
        self.states = states
        n = len(states)
        # Transition matrix: stay in the same state with prob `stickiness`,
        # otherwise spread uniformly across the others.
        self.T = np.full((n, n), (1.0 - stickiness) / (n - 1), dtype=np.float64)
        np.fill_diagonal(self.T, stickiness)
        self.belief = np.full(n, 1.0 / n, dtype=np.float64)

    def update(self, obs: dict[str, float]) -> dict[str, float]:
        obs_vec = np.array([obs.get(s, 0.0) for s in self.states], dtype=np.float64)
        obs_vec = obs_vec + 1e-6  # avoid zero-locking a state
        predicted = self.T.T @ self.belief          # predict step
        posterior = predicted * obs_vec              # measurement update
        posterior /= posterior.sum()
        self.belief = posterior
        return {s: float(p) for s, p in zip(self.states, posterior)}

    def top(self) -> tuple[str, float]:
        i = int(np.argmax(self.belief))
        return self.states[i], float(self.belief[i])


class EmotionDecider:
    """Stable, calibrated emotion decision (replaces the raw HMM for display).

    Two stages:
      1. average the per-frame probabilities over a short window -> steady bars
         (kills frame-to-frame jitter),
      2. a hysteresis state machine commits to a label only when a DIFFERENT
         emotion is clearly stronger (by `switch_margin`) and stays so for `hold`
         frames -> the decision doesn't flicker, but still switches promptly on a
         real, sustained expression.
    """

    def __init__(self, states: list[str], alpha: float = 0.25, switch_margin: float = 0.10,
                 min_conf: float = 0.34, hold: int = 4) -> None:
        self.states = states
        self.alpha = alpha              # EMA: lower = steadier bars, higher = snappier
        self.switch_margin = switch_margin
        self.min_conf = min_conf
        self.hold = hold
        self.ema: np.ndarray | None = None
        self.current = "neutral" if "neutral" in states else states[0]
        self._cand: str | None = None
        self._cand_n = 0

    def update(self, probs: dict[str, float]) -> tuple[str, dict[str, float]]:
        vec = np.array([probs.get(s, 0.0) for s in self.states], dtype=np.float64)
        total = vec.sum()
        if total > 0:
            vec = vec / total
        # EMA smoothing of the probability distribution (steady bars, low jitter).
        self.ema = vec if self.ema is None else self.alpha * vec + (1 - self.alpha) * self.ema
        smoothed = {s: float(self.ema[i]) for i, s in enumerate(self.states)}

        j = int(np.argmax(self.ema))
        top, top_p = self.states[j], float(self.ema[j])
        cur_p = smoothed.get(self.current, 0.0)
        # Hysteresis: commit to a new emotion only when it's clearly stronger than
        # the current one and stays so for `hold` frames -> the label never flickers.
        if top != self.current and top_p >= self.min_conf and (top_p - cur_p) >= self.switch_margin:
            self._cand_n = self._cand_n + 1 if self._cand == top else 1
            self._cand = top
            if self._cand_n >= self.hold:
                self.current = top
                self._cand, self._cand_n = None, 0
        else:
            self._cand, self._cand_n = None, 0
        return self.current, smoothed
