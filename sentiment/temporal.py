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
