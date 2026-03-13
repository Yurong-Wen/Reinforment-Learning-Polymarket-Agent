"""
evaluate.py
───────────
Loads the trained PPO model and runs a full backtest against all four
baseline strategies, then produces:

  • Console table  – mean ROI, Sharpe, Max-DD per agent
  • Plots          – saved to  results/
      ─ backtest_comparison.png   (portfolio curves)
      ─ drawdown_comparison.png   (drawdown profiles)
      ─ return_distribution.png   (box-plots per agent)
      ─ action_distribution.png   (PPO action histogram)

Run:
    python training/evaluate.py
"""

from __future__ import annotations

import os
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend – safe on any machine
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import yaml
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler
from stable_baselines3 import PPO

from data.preprocessing import build_dataset, FEATURE_COLS
from env.polymarket_env import MultiMarketEnv
from agents.baselines import (
    AlwaysBuyYesAgent,
    AlwaysBuyNoAgent,
    MarketOddsAgent,
    RandomAgent,
    run_baseline,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log     = logging.getLogger("evaluate")
console = Console()

os.makedirs("results", exist_ok=True)
RESULTS_DIR = Path("results")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model_and_scaler(model_path: str):
    model        = PPO.load(model_path)
    scaler_path  = model_path + "_scaler.pkl"
    scaler       = None
    if Path(scaler_path).exists():
        with open(scaler_path, "rb") as fh:
            scaler = pickle.load(fh)
        log.info(f"Scaler loaded from {scaler_path}")
    else:
        log.warning("Scaler not found – observations will not be re-scaled.")
    return model, scaler


def make_market_list(X, y, df, market_ids):
    markets = []
    for mid in market_ids:
        rows    = df[df["market_id"] == mid]
        indices = rows.index.tolist()
        if len(indices) < 5:
            continue
        market_X = X[indices]
        outcome  = int(y[indices[0]])
        markets.append((market_X, outcome))
    return markets


def evaluate_ppo(
    model,
    env: MultiMarketEnv,
    n_episodes: int = 50,
    seed: int = 42,
) -> dict:
    """Run PPO agent deterministically for n_episodes; return metrics."""
    rng = np.random.default_rng(seed)
    returns, sharpes, drawdowns = [], [], []
    all_actions   = []
    sample_history = None   # save ONE portfolio history for plotting

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1_000_000)))
        done   = False
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            all_actions.append(int(action))
            done = terminated or truncated

        returns.append(env.total_return())
        sharpes.append(env.sharpe_ratio())
        drawdowns.append(env.max_drawdown())
        if sample_history is None:
            sample_history = list(env.portfolio_history())

    return {
        "agent":          "PPO",
        "returns":        returns,
        "sharpes":        sharpes,
        "drawdowns":      drawdowns,
        "mean_return":    float(np.mean(returns)),
        "mean_sharpe":    float(np.mean(sharpes)),
        "mean_dd":        float(np.mean(drawdowns)),
        "action_counts":  np.bincount(all_actions, minlength=5).tolist(),
        "sample_history": sample_history,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

PALETTE = {
    "PPO":          "#2563eb",   # blue
    "AlwaysBuyYes": "#16a34a",   # green
    "AlwaysBuyNo":  "#dc2626",   # red
    "MarketOdds":   "#d97706",   # amber
    "Random":       "#9ca3af",   # grey
}

ACTION_NAMES = ["HOLD", "BUY_YES", "SELL_YES", "BUY_NO", "SELL_NO"]


def plot_portfolio_curves(results: list[dict], initial_cash: float):
    """One representative portfolio curve per agent."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for r in results:
        hist = r.get("sample_history")
        if hist is None:
            continue
        color = PALETTE.get(r["agent"], "#333333")
        ax.plot(hist, label=r["agent"], color=color, linewidth=1.8)

    ax.axhline(initial_cash, color="black", linestyle="--", linewidth=1,
               label="Starting capital")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Portfolio value ($)")
    ax.set_title("Representative Portfolio Curve per Agent")
    ax.legend(loc="upper left")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("$%.0f"))
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "backtest_comparison.png", dpi=150)
    plt.close(fig)
    log.info("Saved backtest_comparison.png")


def plot_drawdowns(results: list[dict]):
    """Drawdown profile for each agent's sample episode."""
    fig, ax = plt.subplots(figsize=(12, 4))

    for r in results:
        hist = r.get("sample_history")
        if hist is None:
            continue
        arr  = np.array(hist)
        peak = np.maximum.accumulate(arr)
        dd   = (arr - peak) / (peak + 1e-8) * 100
        color = PALETTE.get(r["agent"], "#333333")
        ax.plot(dd, label=r["agent"], color=color, linewidth=1.5)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.fill_between(range(len(dd)), dd, 0,
                    color=PALETTE["PPO"], alpha=0.10, label="_nolegend_")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown Profile")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "drawdown_comparison.png", dpi=150)
    plt.close(fig)
    log.info("Saved drawdown_comparison.png")


def plot_return_distributions(results: list[dict]):
    """Box-plot of episode returns per agent."""
    fig, ax = plt.subplots(figsize=(10, 5))

    labels = [r["agent"]   for r in results]
    data   = [r["returns"] for r in results]
    colors = [PALETTE.get(r["agent"], "#333333") for r in results]

    bp = ax.boxplot(data, patch_artist=True, notch=False, vert=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Episode ROI (%)")
    ax.set_title("Return Distribution Across Episodes")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=1))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "return_distribution.png", dpi=150)
    plt.close(fig)
    log.info("Saved return_distribution.png")


def plot_action_distribution(ppo_result: dict):
    """Bar-chart of PPO action frequency."""
    counts = ppo_result.get("action_counts", [])
    if not counts:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(ACTION_NAMES, counts, color=PALETTE["PPO"], alpha=0.85)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                f"{count:,}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Action")
    ax.set_ylabel("Frequency")
    ax.set_title("PPO Agent — Action Distribution")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "action_distribution.png", dpi=150)
    plt.close(fig)
    log.info("Saved action_distribution.png")


# ── Console Summary Table ─────────────────────────────────────────────────────

def print_summary_table(results: list[dict]):
    table = Table(title="Backtest Summary", show_lines=True)
    table.add_column("Agent",        style="bold")
    table.add_column("Mean ROI (%)", justify="right")
    table.add_column("Mean Sharpe",  justify="right")
    table.add_column("Mean MaxDD (%)", justify="right")

    # Sort by mean return descending
    for r in sorted(results, key=lambda x: x["mean_return"], reverse=True):
        roi_str = f"{r['mean_return']:>+.2f}"
        sharpe  = f"{r['mean_sharpe']:>.3f}"
        dd      = f"{r['mean_dd']:>+.2f}"
        style   = "green" if r["agent"] == "PPO" else "white"
        table.add_row(r["agent"], roi_str, sharpe, dd, style=style)

    console.print(table)

    # Save as CSV
    df = pd.DataFrame([
        {
            "agent":       r["agent"],
            "mean_roi":    r["mean_return"],
            "mean_sharpe": r["mean_sharpe"],
            "mean_maxdd":  r["mean_dd"],
        }
        for r in results
    ])
    df.to_csv(RESULTS_DIR / "summary.csv", index=False)
    log.info("Saved results/summary.csv")


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate(config_path: str = "configs/config.yaml"):
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    data_cfg = cfg["data"]
    env_cfg  = cfg["environment"]
    eval_cfg = cfg["evaluation"]

    # ── 1. Load dataset ───────────────────────────────────────────────────
    log.info("Loading / preprocessing dataset …")
    ds = build_dataset(
        raw_parquet=data_cfg["raw_path"],
        sentiment_path=data_cfg["sentiment_path"],
        processed_path=data_cfg["processed_path"],
        train_ratio=data_cfg["train_ratio"],
    )
    df_test = ds["df_processed"][ds["df_processed"]["split"] == "test"].reset_index(drop=True)
    test_markets = make_market_list(
        ds["X_test"], ds["y_test"], df_test, ds["market_ids_test"]
    )

    if not test_markets:
        log.warning("No test markets – falling back to train markets.")
        df_train = ds["df_processed"][ds["df_processed"]["split"] == "train"].reset_index(drop=True)
        test_markets = make_market_list(
            ds["X_train"], ds["y_train"], df_train, ds["market_ids_train"]
        )

    def _make_env():
        return MultiMarketEnv(
            markets=test_markets,
            initial_cash=env_cfg["initial_cash"],
            max_position_pct=env_cfg["max_position_pct"],
            transaction_cost=env_cfg["transaction_cost"],
            risk_penalty=env_cfg["risk_penalty"],
        )

    n_ep     = eval_cfg["n_episodes"]
    results  = []

    # ── 2. PPO ────────────────────────────────────────────────────────────
    model_path = eval_cfg["model_path"]
    if Path(f"{model_path}.zip").exists():
        model, _ = load_model_and_scaler(model_path)
        env = _make_env()
        ppo_result = evaluate_ppo(model, env, n_episodes=n_ep)
        results.append(ppo_result)
        env.close()
    else:
        log.warning(
            f"Model not found at {model_path}.zip – skipping PPO evaluation.\n"
            "Run training/train.py first."
        )
        ppo_result = None

    # ── 3. Baselines ──────────────────────────────────────────────────────
    baseline_agents = [
        AlwaysBuyYesAgent(),
        AlwaysBuyNoAgent(),
        MarketOddsAgent(),
        RandomAgent(),
    ]

    for agent in baseline_agents:
        env = _make_env()
        r   = run_baseline(agent, env, n_episodes=n_ep)

        # Collect a sample history for plotting
        obs, _ = env.reset(seed=0)
        agent.reset()
        done = False
        while not done:
            action = agent.act(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        r["sample_history"] = list(env.portfolio_history())

        results.append(r)
        env.close()

    # ── 4. Print summary ──────────────────────────────────────────────────
    print_summary_table(results)

    # ── 5. Plots ──────────────────────────────────────────────────────────
    plot_portfolio_curves(results, env_cfg["initial_cash"])
    plot_drawdowns(results)
    plot_return_distributions(results)
    if ppo_result:
        plot_action_distribution(ppo_result)

    console.print(
        "\n[bold green]✓ Evaluation complete.[/]  "
        "Plots saved to [yellow]results/[/]"
    )


if __name__ == "__main__":
    evaluate()
