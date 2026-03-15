"""
train.py
────────
Trains a PPO agent on historical Polymarket data using Stable-Baselines3.

Features
--------
• Vectorised parallel environments (n_envs)
• TensorBoard logging (runs/)
• Periodic checkpoints (models/checkpoints/)
• Best-model auto-save via EvalCallback (models/best/)
• 80 / 20 temporal sample-weighting applied at dataset level

Run:
    python training/train.py
    tensorboard --logdir runs/
"""

from __future__ import annotations

import os
import sys
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from rich.console import Console
from rich.logging import RichHandler
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import SubprocVecEnv

# Allow running as: python training/train.py
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.preprocessing import build_dataset
from env.polymarket_env import MultiMarketEnv

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log     = logging.getLogger("train")
console = Console()


# ── Dataset → (market_data, outcome) list ────────────────────────────────────

def make_market_list(
    X: np.ndarray,
    y: np.ndarray,
    df,
    market_ids: list[str],
) -> list[tuple[np.ndarray, int]]:
    """
    Group the feature matrix back into per-market arrays.

    Returns
    -------
    List of (feature_matrix, outcome) tuples, one per market.
    """
    markets = []
    for current_market_id in market_ids:
        mask    = df[df["market_id"] == current_market_id].index
        # Align the DataFrame index labels with positional indices in X after dropna/reset_index.
        indices = df.index.get_indexer(mask)
        # Remove any -1 entries that indicate a label was not found in the index.
        indices = indices[indices >= 0]
        # Skip markets with fewer than 5 candles because the environment needs a
        # minimum sequence length to compute rolling features without crashing.
        if len(indices) < 5:
            continue
        market_X = X[indices]
        outcome  = int(y[indices[0]])          # The outcome is constant for every row of a given market.
        markets.append((market_X, outcome))
    return markets


# ── Env factory ───────────────────────────────────────────────────────────────

def make_env_fn(markets, env_cfg: dict):
    """Return a callable that creates a monitored MultiMarketEnv."""
    def _init():
        env = MultiMarketEnv(
            markets=markets,
            initial_cash=env_cfg["initial_cash"],
            max_position_pct=env_cfg["max_position_pct"],
            transaction_cost=env_cfg["transaction_cost"],
            risk_penalty=env_cfg["risk_penalty"],
        )
        return Monitor(env)
    return _init


# ── Main Training Function ────────────────────────────────────────────────────

def train(config_path: str = "configs/config.yaml") -> PPO:
    """
    Full training pipeline.

    1. Preprocess raw data (or load cached processed parquet).
    2. Build per-market dataset lists.
    3. Spin up vectorised environments.
    4. Train PPO with callbacks.
    5. Save final model + scaler.
    """

    # ── 1. Load config ────────────────────────────────────────────────────
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    data_cfg  = cfg["data"]
    env_cfg   = cfg["environment"]
    ppo_cfg   = cfg["ppo"]
    eval_cfg  = cfg["evaluation"]

    # ── 2. Preprocess (cached) ────────────────────────────────────────────
    processed_path = data_cfg["processed_path"]

    # Log whether we are reusing a previously cached parquet file or computing
    # features from scratch.  Both cases call build_dataset() — the function
    # itself handles the caching logic internally.
    if Path(processed_path).exists():
        log.info(f"Processed features found at {processed_path} — reusing cache …")
    else:
        log.info("No processed features found — running full preprocessing pipeline …")

    ds = build_dataset(
        raw_parquet=data_cfg["raw_path"],
        sentiment_path=data_cfg["sentiment_path"],
        processed_path=processed_path,
        train_ratio=data_cfg["train_ratio"],
        recent_weight=data_cfg["temporal_weights"]["recent"],
        old_weight=data_cfg["temporal_weights"]["old"],
    )

    X_train = ds["X_train"];  y_train = ds["y_train"]
    X_test  = ds["X_test"];   y_test  = ds["y_test"]
    df_proc = ds["df_processed"]

    # Separate out train/test sub-frames
    df_train = df_proc[df_proc["split"] == "train"].reset_index(drop=True)
    df_test  = df_proc[df_proc["split"] == "test"].reset_index(drop=True)

    # ── 3. Build market lists ─────────────────────────────────────────────
    train_markets = make_market_list(X_train, y_train, df_train, ds["market_ids_train"])
    test_markets  = make_market_list(X_test,  y_test,  df_test,  ds["market_ids_test"])

    console.print(
        f"\n[bold]Markets[/]  train={len(train_markets)}  test={len(test_markets)}\n"
    )
    # A missing training set means the data collection step was never run or
    # all markets were too short and were filtered out.
    if not train_markets:
        raise RuntimeError("No training markets available. Run data collection first.")

    # ── 4. Vectorised training environments ───────────────────────────────
    # Cap the number of parallel workers at the number of available markets so
    # that each worker is guaranteed at least one market to sample from.
    number_of_parallel_environments = min(ppo_cfg["n_envs"], len(train_markets))
    vec_env = make_vec_env(
        make_env_fn(train_markets, env_cfg),
        n_envs=number_of_parallel_environments,
        vec_env_cls=SubprocVecEnv if number_of_parallel_environments > 1 else None,
    )

    # ── 5. Evaluation environment ─────────────────────────────────────────
    eval_env = Monitor(
        MultiMarketEnv(
            markets=test_markets if test_markets else train_markets,
            initial_cash=env_cfg["initial_cash"],
            max_position_pct=env_cfg["max_position_pct"],
            transaction_cost=env_cfg["transaction_cost"],
            risk_penalty=env_cfg["risk_penalty"],
        )
    )

    # ── 6. PPO model ──────────────────────────────────────────────────────
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=ppo_cfg["learning_rate"],
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_range=ppo_cfg["clip_range"],
        ent_coef=ppo_cfg["ent_coef"],
        vf_coef=ppo_cfg["vf_coef"],
        max_grad_norm=ppo_cfg["max_grad_norm"],
        verbose=1,
        tensorboard_log="./runs/",
        device="auto",
    )

    # ── 7. Callbacks ──────────────────────────────────────────────────────
    os.makedirs("models/best",         exist_ok=True)
    os.makedirs("models/checkpoints",  exist_ok=True)

    callbacks = CallbackList([
        # EvalCallback runs the agent on the held-out evaluation environment at regular
        # intervals and saves the model weights whenever a new best mean reward is achieved.
        EvalCallback(
            eval_env,
            best_model_save_path="models/best/",
            log_path="models/best/",
            eval_freq=max(ppo_cfg["n_steps"] * 2, 10_000),
            n_eval_episodes=10,
            deterministic=True,
            render=False,
        ),
        # CheckpointCallback saves a periodic snapshot regardless of reward, providing
        # recovery points if training is interrupted before the best model is saved.
        CheckpointCallback(
            save_freq=50_000,
            save_path="models/checkpoints/",
            name_prefix="ppo_poly",
        ),
    ])

    # ── 8. Train ──────────────────────────────────────────────────────────
    console.print(
        f"\n[bold green]Starting PPO training[/]  "
        f"total_timesteps={ppo_cfg['total_timesteps']:,}  "
        f"n_envs={number_of_parallel_environments}\n"
    )
    model.learn(
        total_timesteps=ppo_cfg["total_timesteps"],
        callback=callbacks,
        progress_bar=True,
    )

    # ── 9. Save final model + scaler ──────────────────────────────────────
    final_path = eval_cfg["model_path"]
    os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
    model.save(final_path)

    scaler_path = final_path + "_scaler.pkl"
    with open(scaler_path, "wb") as fh:
        pickle.dump(ds["scaler"], fh)

    console.print(
        f"\n[bold green]✓ Training complete[/]\n"
        f"  Model  → [yellow]{final_path}.zip[/]\n"
        f"  Scaler → [yellow]{scaler_path}[/]"
    )
    vec_env.close()
    eval_env.close()
    return model


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
