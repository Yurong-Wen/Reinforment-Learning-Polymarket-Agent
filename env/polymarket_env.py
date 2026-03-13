"""
polymarket_env.py
─────────────────
Custom OpenAI Gymnasium environment that models binary prediction-market
trading as a Partially Observable Markov Decision Process (POMDP).

┌──────────────────────────────────────────────────────────┐
│  POMDP  Formulation                                      │
│  ─────────────────                                       │
│  State   – true outcome probability (HIDDEN)             │
│  Obs     – market price features + position info         │
│  Actions – HOLD / BUY_YES / SELL_YES / BUY_NO / SELL_NO │
│  Reward  – step P&L − risk-concentration penalty         │
│  Episode – one resolved market (start → resolution)      │
└──────────────────────────────────────────────────────────┘

Each episode replays a SINGLE resolved market.  Multiple markets are
handled by wrapping this env in VecEnv (or by cycling markets in reset()).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

log = logging.getLogger(__name__)

# ── Action constants ──────────────────────────────────────────────────────────
HOLD     = 0
BUY_YES  = 1
SELL_YES = 2
BUY_NO   = 3
SELL_NO  = 4
ACTION_LABELS = {
    HOLD: "HOLD", BUY_YES: "BUY_YES", SELL_YES: "SELL_YES",
    BUY_NO: "BUY_NO", SELL_NO: "SELL_NO",
}

# Index of the yes_prob feature inside the feature vector
YES_PROB_IDX = 6   # must match FEATURE_COLS order in preprocessing.py


class PolymarketEnv(gym.Env):
    """
    Binary prediction-market trading environment.

    Parameters
    ----------
    market_data : np.ndarray, shape (T, n_features)
        Pre-scaled feature matrix for ONE resolved market.
        Rows are time-ordered OHLCV candles.
    outcome : int
        Binary resolution: 1 = YES won, 0 = NO won.
    initial_cash : float
        Starting capital in USD.
    max_position_pct : float
        Maximum fraction of portfolio value to bet per action.
    transaction_cost : float
        Proportional fee deducted on each buy / sell (e.g. 0.002 = 0.2 %).
    risk_penalty : float
        Coefficient λ for the concentration-penalty term in the reward.
    render_mode : str or None
        "human" prints a one-line summary each step.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        market_data:       np.ndarray,
        outcome:           int,
        initial_cash:      float = 1_000.0,
        max_position_pct:  float = 0.30,
        transaction_cost:  float = 0.002,
        risk_penalty:      float = 0.01,
        render_mode:       Optional[str] = None,
    ):
        super().__init__()

        assert market_data.ndim == 2, "market_data must be 2-D (T × n_features)"
        assert outcome in (0, 1),     "outcome must be 0 (No) or 1 (Yes)"

        self.data         = market_data.astype(np.float32)
        self.outcome      = int(outcome)
        self.T            = len(self.data)
        self.n_features   = self.data.shape[1]

        self.initial_cash     = float(initial_cash)
        self.max_pos_pct      = float(max_position_pct)
        self.tc               = float(transaction_cost)
        self.risk_penalty     = float(risk_penalty)
        self.render_mode      = render_mode

        # Observation = market features + 4 portfolio features
        # [yes_pos_ratio, no_pos_ratio, cash_ratio, time_remaining]
        n_obs = self.n_features + 4
        self.observation_space = spaces.Box(
            low  = -np.inf,
            high =  np.inf,
            shape= (n_obs,),
            dtype= np.float32,
        )

        # 5 discrete actions
        self.action_space = spaces.Discrete(5)

        # Episode state (initialised in reset)
        self.t          = 0
        self.cash       = self.initial_cash
        self.yes_shares = 0.0
        self.no_shares  = 0.0
        self.portfolio_history: list[float] = []
        self.action_history:    list[int]   = []
        self.reward_history:    list[float] = []

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _yes_price(self) -> float:
        """Current implied YES price (probability), clipped to (0.01, 0.99)."""
        raw = float(self.data[self.t, YES_PROB_IDX])
        return float(np.clip(raw, 0.01, 0.99))

    def _portfolio_value(self, yes_price: Optional[float] = None) -> float:
        """Total mark-to-market portfolio value."""
        yp = yes_price if yes_price is not None else self._yes_price()
        np_ = 1.0 - yp
        return self.cash + self.yes_shares * yp + self.no_shares * np_

    def _get_obs(self) -> np.ndarray:
        """Concatenate market features with portfolio state."""
        yp  = self._yes_price()
        np_ = 1.0 - yp
        pv  = self._portfolio_value(yp)

        portfolio_feats = np.array(
            [
                self.yes_shares * yp  / (pv + 1e-8),   # YES position ratio
                self.no_shares  * np_ / (pv + 1e-8),   # NO  position ratio
                self.cash             / (pv + 1e-8),   # cash ratio
                1.0 - self.t / (self.T - 1 + 1e-8),    # time remaining [0,1]
            ],
            dtype=np.float32,
        )
        return np.concatenate([self.data[self.t], portfolio_feats])

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self,
        seed:    Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.t          = 0
        self.cash       = self.initial_cash
        self.yes_shares = 0.0
        self.no_shares  = 0.0
        self.portfolio_history = [self.initial_cash]
        self.action_history    = []
        self.reward_history    = []

        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one trading step.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        yp  = self._yes_price()
        np_ = 1.0 - yp
        pv_before = self._portfolio_value(yp)
        bet_size  = self.max_pos_pct * pv_before

        # ── Execute trade ─────────────────────────────────────────────────
        if action == BUY_YES and self.cash >= bet_size * 0.99:
            shares          = (bet_size * (1.0 - self.tc)) / yp
            self.yes_shares += shares
            self.cash       -= bet_size

        elif action == SELL_YES and self.yes_shares > 1e-9:
            proceeds         = self.yes_shares * yp * (1.0 - self.tc)
            self.cash       += proceeds
            self.yes_shares  = 0.0

        elif action == BUY_NO and self.cash >= bet_size * 0.99:
            shares          = (bet_size * (1.0 - self.tc)) / np_
            self.no_shares += shares
            self.cash      -= bet_size

        elif action == SELL_NO and self.no_shares > 1e-9:
            proceeds        = self.no_shares * np_ * (1.0 - self.tc)
            self.cash      += proceeds
            self.no_shares  = 0.0
        # HOLD: no change

        # ── Advance time ──────────────────────────────────────────────────
        self.t += 1
        terminated = self.t >= self.T - 1

        # ── Final resolution payoff ───────────────────────────────────────
        if terminated:
            # YES shares pay $1 each if YES won, else $0
            yes_payoff = self.yes_shares * (1.0 if self.outcome == 1 else 0.0)
            # NO  shares pay $1 each if NO  won, else $0
            no_payoff  = self.no_shares  * (1.0 if self.outcome == 0 else 0.0)
            self.cash       += yes_payoff + no_payoff
            self.yes_shares  = 0.0
            self.no_shares   = 0.0

        # ── Compute reward ────────────────────────────────────────────────
        yp_new  = self._yes_price() if not terminated else yp
        np_new_ = 1.0 - yp_new
        pv_after = self._portfolio_value(yp_new if not terminated else 1.0)

        # Step P&L (log return for numerical stability)
        pnl = (pv_after - pv_before) / (pv_before + 1e-8)

        # Concentration penalty encourages diversification
        concentration = (
            abs(self.yes_shares * yp_new)  +
            abs(self.no_shares  * np_new_)
        ) / (pv_after + 1e-8)

        reward = float(pnl - self.risk_penalty * concentration)

        # ── Book-keeping ──────────────────────────────────────────────────
        self.portfolio_history.append(pv_after)
        self.action_history.append(action)
        self.reward_history.append(reward)

        obs  = self._get_obs()
        info = {
            "portfolio_value": pv_after,
            "pnl":             pnl,
            "action_label":    ACTION_LABELS[action],
            "yes_price":       yp,
            "yes_shares":      self.yes_shares,
            "no_shares":       self.no_shares,
            "cash":            self.cash,
            "step":            self.t,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, False, info

    # ── Metrics helpers ───────────────────────────────────────────────────────

    def total_return(self) -> float:
        """Percentage return over the episode."""
        hist = self.portfolio_history
        if len(hist) < 2:
            return 0.0
        return (hist[-1] / hist[0] - 1.0) * 100.0

    def sharpe_ratio(self, periods_per_year: int = 8760) -> float:
        """Annualised Sharpe Ratio (hourly candles → 8760 periods/year)."""
        hist  = np.array(self.portfolio_history)
        rets  = np.diff(hist) / (hist[:-1] + 1e-8)
        if rets.std() < 1e-10:
            return 0.0
        return float((rets.mean() / rets.std()) * np.sqrt(periods_per_year))

    def max_drawdown(self) -> float:
        """Maximum peak-to-trough drawdown as a percentage."""
        hist = np.array(self.portfolio_history)
        peak = np.maximum.accumulate(hist)
        dd   = (hist - peak) / (peak + 1e-8)
        return float(dd.min() * 100.0)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode != "human":
            return
        pv = self._portfolio_value()
        ret = (pv / self.initial_cash - 1.0) * 100.0
        last_action = ACTION_LABELS.get(
            self.action_history[-1] if self.action_history else HOLD
        )
        print(
            f"[Step {self.t:>4d}/{self.T}] "
            f"PV={pv:>8.2f}  ret={ret:>+6.2f}%  "
            f"YES={self.yes_shares:.3f}  NO={self.no_shares:.3f}  "
            f"Cash={self.cash:.2f}  Action={last_action}"
        )

    def close(self):
        pass


# ── Multi-market wrapper ──────────────────────────────────────────────────────

class MultiMarketEnv(gym.Env):
    """
    Wraps a collection of (market_data, outcome) pairs.
    Each episode randomly samples ONE resolved market and replays it.

    This is the env you pass to stable-baselines3 VecEnv.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        markets:          list[tuple[np.ndarray, int]],
        initial_cash:     float = 1_000.0,
        max_position_pct: float = 0.30,
        transaction_cost: float = 0.002,
        risk_penalty:     float = 0.01,
        render_mode:      Optional[str] = None,
    ):
        super().__init__()
        assert markets, "markets list must not be empty"
        self.markets = markets

        # Infer observation size from first market
        sample_data, sample_outcome = markets[0]
        self._env = PolymarketEnv(
            sample_data, sample_outcome,
            initial_cash, max_position_pct, transaction_cost,
            risk_penalty, render_mode,
        )

        self.observation_space = self._env.observation_space
        self.action_space      = self._env.action_space
        self._current_idx      = 0

    def reset(self, seed=None, options=None):
        # Sample a random market each episode
        rng = np.random.default_rng(seed)
        self._current_idx = int(rng.integers(len(self.markets)))
        data, outcome = self.markets[self._current_idx]

        self._env = PolymarketEnv(
            data, outcome,
            self._env.initial_cash,
            self._env.max_pos_pct,
            self._env.tc,
            self._env.risk_penalty,
            self._env.render_mode,
        )
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        return self._env.step(action)

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()

    # Expose episode metrics from the inner env
    def total_return(self)  -> float: return self._env.total_return()
    def sharpe_ratio(self)  -> float: return self._env.sharpe_ratio()
    def max_drawdown(self)  -> float: return self._env.max_drawdown()
    def portfolio_history(self) -> list: return self._env.portfolio_history
