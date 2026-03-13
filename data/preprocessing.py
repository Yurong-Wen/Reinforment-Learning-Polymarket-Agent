"""
preprocessing.py
────────────────
Turns raw Polymarket OHLCV data into a feature matrix ready for the RL
environment.  Key responsibilities:

1. Technical feature engineering  (momentum, volatility, micro-structure)
2. Sentiment merge                (FinBERT scores per market)
3. 80 / 20 temporal weighting     (recent ≫ old)
4. Train / test split             (by market, not by row)
5. Per-split z-score scaling

Run standalone:
    python data/preprocessing.py
"""

import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from rich.logging import RichHandler
from rich.console import Console

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log     = logging.getLogger("preprocessing")
console = Console()

# ── Feature columns used by the Gym environment ───────────────────────────────
FEATURE_COLS = [
    "ret_1h",
    "ret_6h",
    "ret_24h",
    "vol_24h",
    "vol_ratio",
    "spread_proxy",
    "yes_prob",
    "no_prob",
    "kelly_yes",
    "kelly_no",
    "sentiment_score",
]


# ── Technical Feature Engineering ─────────────────────────────────────────────

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-market technical indicators.
    `df` must have columns: t, c, h, l, volume, market_id
    """
    df = df.copy().sort_values(["market_id", "t"])

    grp = df.groupby("market_id", sort=False)

    # ── Price momentum ──────────────────────────────────────────────────
    df["ret_1h"]  = grp["c"].transform(lambda x: x.pct_change(1))
    df["ret_6h"]  = grp["c"].transform(lambda x: x.pct_change(6))
    df["ret_24h"] = grp["c"].transform(lambda x: x.pct_change(24))

    # ── Rolling volatility ──────────────────────────────────────────────
    df["vol_24h"] = grp["ret_1h"].transform(lambda x: x.rolling(24, min_periods=4).std())

    # ── Volume features ─────────────────────────────────────────────────
    df["vol_ma_6h"] = grp["volume"].transform(lambda x: x.rolling(6, min_periods=1).mean())
    df["vol_ratio"] = df["volume"] / (df["vol_ma_6h"] + 1e-8)

    # ── Market micro-structure (proxy bid-ask spread) ───────────────────
    df["spread_proxy"] = (df["h"] - df["l"]) / (df["c"].clip(lower=1e-8))

    # ── Implied probability (clipped to avoid log issues) ───────────────
    df["yes_prob"] = df["c"].clip(0.02, 0.98)
    df["no_prob"]  = 1.0 - df["yes_prob"]

    # ── Kelly criterion signals ─────────────────────────────────────────
    #   Expected value: p_win * 1 - p_lose * 1   (binary payoff)
    df["kelly_yes"] = df["yes_prob"] - df["no_prob"]
    df["kelly_no"]  = df["no_prob"]  - df["yes_prob"]

    return df


# ── Sentiment Merge ───────────────────────────────────────────────────────────

def merge_sentiment(
    df:           pd.DataFrame,
    sentiment_path: str = "data/sentiment/sentiment_scores.parquet",
) -> pd.DataFrame:
    """Left-join per-market sentiment scores; fill missing with 0."""
    if not Path(sentiment_path).exists():
        log.warning(f"Sentiment file not found at {sentiment_path} – using 0.")
        df["sentiment_score"] = 0.0
        return df

    sent = pd.read_parquet(sentiment_path)[["market_id", "sentiment_score"]]
    df = df.merge(sent, on="market_id", how="left")
    df["sentiment_score"] = df["sentiment_score"].fillna(0.0)
    return df


# ── 80 / 20 Temporal Weighting ────────────────────────────────────────────────

def compute_temporal_weights(
    n:             int,
    recent_weight: float = 0.80,
    old_weight:    float = 0.20,
) -> np.ndarray:
    """
    Assign per-row sample weights so that:
      - The *older* half  of timesteps carries `old_weight`    (default 20 %)
      - The *recent* half of timesteps carries `recent_weight` (default 80 %)

    Returns a normalised weight array of shape (n,).

    This implements the project's stated weighting policy:
    "greater emphasis on recent observations (80%) more than older data (20%)"
    """
    weights = np.zeros(n)
    split   = n // 2

    if split > 0:
        weights[:split] = old_weight    / split           # older  half
    if n - split > 0:
        weights[split:] = recent_weight / (n - split)     # recent half

    weights /= weights.sum()                              # normalise to sum = 1
    return weights.astype(np.float32)


# ── Full Preprocessing Pipeline ───────────────────────────────────────────────

def build_dataset(
    raw_parquet:      str   = "data/raw/polymarket_raw.parquet",
    sentiment_path:   str   = "data/sentiment/sentiment_scores.parquet",
    processed_path:   str   = "data/processed/features.parquet",
    train_ratio:      float = 0.80,
    recent_weight:    float = 0.80,
    old_weight:       float = 0.20,
) -> dict:
    """
    Full preprocessing pipeline.

    Returns
    -------
    dict with keys:
      X_train, y_train, w_train   – training feature matrix, labels, weights
      X_test,  y_test,  w_test    – held-out feature matrix, labels, weights
      scaler                       – fitted StandardScaler (for live inference)
      market_ids_train/test        – list of market IDs in each split
      df_processed                 – full processed DataFrame (for EDA)
    """
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)

    # 1. Load raw data
    log.info(f"Loading raw data from {raw_parquet} …")
    df = pd.read_parquet(raw_parquet)
    log.info(f"  Raw shape: {df.shape}")

    # 2. Drop markets with unknown outcome
    df = df[df["outcome"].isin([0, 1])].copy()
    log.info(f"  After dropping unknown outcomes: {df.shape}")

    # 3. Technical features
    log.info("Computing technical features …")
    df = add_technical_features(df)

    # 4. Merge sentiment
    log.info("Merging sentiment scores …")
    df = merge_sentiment(df, sentiment_path)

    # 5. Drop NaN rows (from rolling windows)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    log.info(f"  After dropna: {df.shape}")

    # 6. Train / test split BY MARKET (no data leakage)
    all_markets = df["market_id"].unique()
    np.random.seed(42)
    np.random.shuffle(all_markets)
    n_train = int(len(all_markets) * train_ratio)
    train_markets = all_markets[:n_train]
    test_markets  = all_markets[n_train:]

    df_train = df[df["market_id"].isin(train_markets)].copy()
    df_test  = df[df["market_id"].isin(test_markets)].copy()
    log.info(
        f"  Train: {len(train_markets)} markets / {len(df_train):,} rows  |  "
        f"Test:  {len(test_markets)}  markets / {len(df_test):,}  rows"
    )

    # 7. Feature matrices
    X_train_raw = df_train[FEATURE_COLS].values.astype(np.float32)
    X_test_raw  = df_test [FEATURE_COLS].values.astype(np.float32)

    # 8. Z-score scaling (fit on train only)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test  = scaler.transform(X_test_raw).astype(np.float32)

    # 9. Per-row temporal weights (80 / 20)
    w_train = compute_temporal_weights(len(X_train), recent_weight, old_weight)
    w_test  = compute_temporal_weights(len(X_test),  recent_weight, old_weight)

    # 10. Labels (binary outcome per market row)
    y_train = df_train["outcome"].values.astype(np.int8)
    y_test  = df_test ["outcome"].values.astype(np.int8)

    # 11. Save processed DataFrame for EDA / notebooks
    df_train["split"] = "train"
    df_test ["split"] = "test"
    df_processed = pd.concat([df_train, df_test], ignore_index=True)
    df_processed.to_parquet(processed_path, index=False)
    log.info(f"  Saved processed features → {processed_path}")

    result = dict(
        X_train=X_train, y_train=y_train, w_train=w_train,
        X_test =X_test,  y_test =y_test,  w_test =w_test,
        scaler=scaler,
        market_ids_train=train_markets.tolist(),
        market_ids_test =test_markets.tolist(),
        df_processed=df_processed,
        feature_cols=FEATURE_COLS,
    )
    console.print(
        "\n[bold green]✓ Preprocessing complete.[/]\n"
        f"  Train: [cyan]{X_train.shape}[/] | "
        f"Test: [cyan]{X_test.shape}[/]\n"
        f"  Features: {FEATURE_COLS}"
    )
    return result


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml

    with open("configs/config.yaml") as fh:
        cfg = yaml.safe_load(fh)

    d = cfg["data"]
    build_dataset(
        raw_parquet=d["raw_path"],
        sentiment_path=d["sentiment_path"],
        processed_path=d["processed_path"],
        train_ratio=d["train_ratio"],
        recent_weight=d["temporal_weights"]["recent"],
        old_weight=d["temporal_weights"]["old"],
    )
