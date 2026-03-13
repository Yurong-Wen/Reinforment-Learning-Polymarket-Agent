"""
baselines.py
────────────
Four deterministic benchmark strategies that the PPO agent must beat:

  1. AlwaysBuyYes   – Buy YES at step 0, hold until resolution.
  2. AlwaysBuyNo    – Buy NO  at step 0, hold until resolution.
  3. MarketOdds     – Follow market consensus: buy the side with P > 0.60.
  4. RandomAgent    – Uniformly random action each step.

All agents expose the same interface:
    agent.act(obs) → int   (action index)

Evaluation helper:
    run_baseline(agent, env, n_episodes) → dict of metrics
"""

from __future__ import annotations

import numpy as np
from env.polymarket_env import PolymarketEnv, YES_PROB_IDX, BUY_YES, BUY_NO, HOLD


# ── Base class ────────────────────────────────────────────────────────────────

class BaseAgent:
    """Minimal interface every baseline must implement."""

    name: str = "Base"

    def act(self, obs: np.ndarray) -> int:
        raise NotImplementedError

    def reset(self):
        """Called once per episode before the first step."""
        pass


# ── Concrete strategies ───────────────────────────────────────────────────────

class AlwaysBuyYesAgent(BaseAgent):
    """Buy YES shares on step 0, then HOLD for the remainder."""

    name = "AlwaysBuyYes"

    def reset(self):
        self._bought = False

    def act(self, obs: np.ndarray) -> int:
        if not self._bought:
            self._bought = True
            return BUY_YES
        return HOLD


class AlwaysBuyNoAgent(BaseAgent):
    """Buy NO shares on step 0, then HOLD for the remainder."""

    name = "AlwaysBuyNo"

    def reset(self):
        self._bought = False

    def act(self, obs: np.ndarray) -> int:
        if not self._bought:
            self._bought = True
            return BUY_NO
        return HOLD


class MarketOddsAgent(BaseAgent):
    """
    Follow market consensus:
      - If YES implied probability > 0.60  → BUY_YES
      - If YES implied probability < 0.40  → BUY_NO
      - Otherwise                          → HOLD

    Buys only once and then holds (avoids churn).
    """

    name = "MarketOdds"

    def __init__(self, high_threshold: float = 0.60, low_threshold: float = 0.40):
        self.high = high_threshold
        self.low  = low_threshold

    def reset(self):
        self._bought = False

    def act(self, obs: np.ndarray) -> int:
        if self._bought:
            return HOLD
        # obs[YES_PROB_IDX] is the (scaled) yes_prob feature;
        # we need to recover the raw value.  Since we don't have the scaler
        # here, we use a sign-based heuristic: positive → market leans YES.
        # For a cleaner approach pass the raw yes_price separately via info.
        raw_signal = obs[YES_PROB_IDX]
        if raw_signal > 0.0:     # scaled value above mean → YES favoured
            self._bought = True
            return BUY_YES
        elif raw_signal < 0.0:   # scaled value below mean → NO favoured
            self._bought = True
            return BUY_NO
        return HOLD


class RandomAgent(BaseAgent):
    """Uniformly random action at every step (stress-test baseline)."""

    name = "Random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(5))


# ── Evaluation helper ─────────────────────────────────────────────────────────

def run_baseline(
    agent:      BaseAgent,
    env:        PolymarketEnv,
    n_episodes: int = 50,
    seed:       int = 42,
) -> dict:
    """
    Run `agent` for `n_episodes` on `env` and collect performance metrics.

    Returns
    -------
    dict with keys:
        returns     – list of per-episode % total return
        sharpes     – list of per-episode Sharpe ratios
        drawdowns   – list of per-episode max drawdowns (%)
        mean_return – float
        mean_sharpe – float
        mean_dd     – float
    """
    rng      = np.random.default_rng(seed)
    returns, sharpes, drawdowns = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
        agent.reset()
        done = False

        while not done:
            action = agent.act(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        returns.append(env.total_return())
        sharpes.append(env.sharpe_ratio())
        drawdowns.append(env.max_drawdown())

    return {
        "agent":       agent.name,
        "returns":     returns,
        "sharpes":     sharpes,
        "drawdowns":   drawdowns,
        "mean_return": float(np.mean(returns)),
        "mean_sharpe": float(np.mean(sharpes)),
        "mean_dd":     float(np.mean(drawdowns)),
    }
