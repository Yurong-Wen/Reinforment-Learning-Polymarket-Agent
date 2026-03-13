"""
dashboard/app.py
────────────────
Streamlit live paper-trading dashboard for the Polymarket RL agent.

Tabs
────
  📊 Overview        – key KPIs + portfolio equity curve
  🤖 Agent Live      – step-through the agent on a market in real time
  📈 Backtest        – load pre-computed backtest results
  📰 Market Browser  – browse resolved markets from the dataset

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sys
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yaml

# Allow imports from project root regardless of working directory
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Polymarket RL Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ───────────────────────────────────────────────────────────────────
CONFIG_PATH = ROOT / "configs" / "config.yaml"

@st.cache_data(show_spinner=False)
def load_config():
    with open(CONFIG_PATH) as fh:
        return yaml.safe_load(fh)

@st.cache_resource(show_spinner="Loading PPO model …")
def load_model(model_path: str):
    p = Path(model_path)
    candidates = []

    # Primary target from config/sidebar.
    candidates.append(p)
    if p.suffix != ".zip":
        candidates.append(p.with_suffix(".zip"))

    # Fallbacks when training did not finish final save.
    candidates.append(ROOT / "models" / "best" / "best_model.zip")
    ckpts = sorted(
        (ROOT / "models" / "checkpoints").glob("ppo_poly_*_steps.zip"),
        key=lambda fp: fp.stat().st_mtime,
        reverse=True,
    )
    if ckpts:
        candidates.append(ckpts[0])

    resolved = next((c for c in candidates if c.exists()), None)
    if resolved is None:
        st.warning(
            "No PPO model file found. Tried configured path, "
            "`models/best/best_model.zip`, and latest checkpoint."
        )
        return None

    try:
        from stable_baselines3 import PPO
        return PPO.load(str(resolved))
    except Exception as exc:
        st.warning(f"Could not load model: {exc}")
        return None

@st.cache_data(show_spinner="Loading dataset …")
def load_processed(_path: str):
    p = Path(_path)
    if not p.exists():
        return None
    return pd.read_parquet(p)

@st.cache_data(show_spinner=False)
def load_summary(_path: str):
    p = Path(_path)
    if not p.exists():
        return None
    return pd.read_csv(p)


def metric_delta_color(val: float) -> str:
    return "normal" if val >= 0 else "inverse"


def add_question_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a human-readable question label column with deduplication.

    If multiple markets share the same question text, append " (2)", " (3)", ...
    so select boxes remain unambiguous without exposing market IDs.
    """
    out = df.copy()
    seen = {}
    labels = []
    for raw in out["question"].fillna("Untitled market").astype(str):
        base = raw.strip() or "Untitled market"
        idx = seen.get(base, 0) + 1
        seen[base] = idx
        labels.append(base if idx == 1 else f"{base} ({idx})")
    out["display_question"] = labels
    return out


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://polymarket.com/favicon.ico", width=32)
    st.title("Polymarket RL")
    st.caption("Reinforcement Learning on Prediction Markets")
    st.divider()

    cfg         = load_config()
    model_path  = st.text_input("Model path", value=cfg["evaluation"]["model_path"])
    initial_cash = st.number_input(
        "Starting capital ($)", min_value=10.0, max_value=100_000.0,
        value=float(cfg["environment"]["initial_cash"]), step=100.0,
    )
    max_pos = st.slider(
        "Max position %", min_value=0.05, max_value=0.50,
        value=float(cfg["environment"]["max_position_pct"]), step=0.05,
    )
    st.divider()
    st.markdown("**Data paths**")
    processed_path = st.text_input(
        "Processed features", value=str(ROOT / cfg["data"]["processed_path"])
    )
    results_csv = str(ROOT / "results" / "summary.csv")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_live, tab_backtest, tab_browser = st.tabs(
    ["📊 Overview", "🤖 Agent Live", "📈 Backtest Results", "📰 Market Browser"]
)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)
    summary = load_summary(results_csv)
    if summary is not None and not summary.empty:
        ppo_row = summary[summary["agent"] == "PPO"]
        best_base = summary[summary["agent"] != "PPO"].sort_values(
            "mean_roi", ascending=False
        ).iloc[0] if len(summary) > 1 else None

        ppo_roi    = float(ppo_row["mean_roi"].iloc[0])    if not ppo_row.empty else 0.0
        ppo_sharpe = float(ppo_row["mean_sharpe"].iloc[0]) if not ppo_row.empty else 0.0
        base_roi   = float(best_base["mean_roi"])          if best_base is not None else 0.0

        col1.metric("PPO Mean ROI",    f"{ppo_roi:+.2f}%",  f"vs best baseline {ppo_roi - base_roi:+.2f}%")
        col2.metric("PPO Sharpe",      f"{ppo_sharpe:.3f}")
        col3.metric("Best Baseline",   f"{base_roi:+.2f}%", str(best_base["agent"]) if best_base is not None else "—")

        st.subheader("Agent Comparison")
        fig = px.bar(
            summary.sort_values("mean_roi", ascending=False),
            x="agent", y="mean_roi",
            color="agent",
            color_discrete_map={
                "PPO": "#2563eb", "AlwaysBuyYes": "#16a34a",
                "AlwaysBuyNo": "#dc2626", "MarketOdds": "#d97706",
                "Random": "#9ca3af",
            },
            labels={"mean_roi": "Mean ROI (%)", "agent": "Agent"},
            title="Mean Episode ROI by Agent",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "No backtest results yet.  "
            "Run `python training/evaluate.py` to generate `results/summary.csv`."
        )

    # Static images from results/
    img_dir = ROOT / "results"
    images  = list(img_dir.glob("*.png"))
    if images:
        st.subheader("Backtest Plots")
        cols = st.columns(2)
        for i, img in enumerate(sorted(images)):
            cols[i % 2].image(str(img), caption=img.stem.replace("_", " ").title())


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Agent Live Step-Through
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.header("🤖 Live Agent Step-Through")
    st.caption(
        "Load a processed market, then step through it tick-by-tick "
        "watching the PPO agent trade in real time."
    )

    df_proc = load_processed(processed_path)
    model   = load_model(str(ROOT / model_path))

    if df_proc is None:
        st.warning(
            "Processed features not found.  "
            "Run `python data/preprocessing.py` first."
        )
    else:
        from data.preprocessing import FEATURE_COLS

        # Market selector (question-first display, id used internally)
        market_meta = (
            df_proc.groupby("market_id", as_index=False)
            .agg(question=("question", "first"), n_candles=("t", "count"))
            .sort_values("question")
            .reset_index(drop=True)
        )
        market_meta = add_question_labels(market_meta)
        sel_label = st.selectbox(
            "Select market question",
            market_meta["display_question"].tolist(),
        )
        sel_market = market_meta.loc[
            market_meta["display_question"] == sel_label, "market_id"
        ].iloc[0]
        mkt_df = df_proc[df_proc["market_id"] == sel_market].sort_values("t").reset_index(drop=True)
        outcome = int(mkt_df["outcome"].iloc[0])

        st.markdown(
            f"**Question:** {mkt_df['question'].iloc[0]}  \n"
            f"**Outcome:** {'✅ YES' if outcome == 1 else '❌ NO'}  \n"
            f"**Candles:** {len(mkt_df)}"
        )

        # Build feature matrix
        feat_cols_present = [c for c in FEATURE_COLS if c in mkt_df.columns]
        X = mkt_df[feat_cols_present].fillna(0).values.astype(np.float32)

        # Session state for the episode
        if "live_step" not in st.session_state or st.session_state.get("live_market") != sel_market:
            st.session_state.live_step      = 0
            st.session_state.live_cash      = initial_cash
            st.session_state.live_yes       = 0.0
            st.session_state.live_no        = 0.0
            st.session_state.live_portfolio = [initial_cash]
            st.session_state.live_actions   = []
            st.session_state.live_market    = sel_market

        # Controls
        c1, c2, c3 = st.columns([1, 1, 2])
        step_btn  = c1.button("▶ Next Step")
        run10_btn = c2.button("⏩ Run 10 Steps")
        reset_btn = c3.button("🔄 Reset Episode")

        if reset_btn:
            st.session_state.live_step      = 0
            st.session_state.live_cash      = initial_cash
            st.session_state.live_yes       = 0.0
            st.session_state.live_no        = 0.0
            st.session_state.live_portfolio = [initial_cash]
            st.session_state.live_actions   = []

        ACTION_LABELS = ["HOLD", "BUY_YES", "SELL_YES", "BUY_NO", "SELL_NO"]
        ACTION_COLORS = {
            "HOLD": "grey", "BUY_YES": "green", "SELL_YES": "lightgreen",
            "BUY_NO": "red", "SELL_NO": "salmon",
        }

        def _do_step(n_steps=1):
            from env.polymarket_env import PolymarketEnv, YES_PROB_IDX
            # Reconstruct env at current state is complex; use inner env directly
            # Simplified: just show the PPO action given the current observation
            for _ in range(n_steps):
                t = st.session_state.live_step
                if t >= len(X) - 1:
                    break
                yp   = float(np.clip(X[t, YES_PROB_IDX if YES_PROB_IDX < X.shape[1] else 0], 0.01, 0.99))
                np_  = 1.0 - yp
                pv   = (st.session_state.live_cash
                        + st.session_state.live_yes  * yp
                        + st.session_state.live_no   * np_)

                portfolio_feats = np.array([
                    st.session_state.live_yes  * yp  / (pv + 1e-8),
                    st.session_state.live_no   * np_ / (pv + 1e-8),
                    st.session_state.live_cash        / (pv + 1e-8),
                    1.0 - t / (len(X) - 1 + 1e-8),
                ], dtype=np.float32)
                obs = np.concatenate([X[t], portfolio_feats])

                if model is not None:
                    action, _ = model.predict(obs, deterministic=True)
                    action    = int(action)
                else:
                    action = 0  # HOLD if no model

                # Apply action
                bet = max_pos * pv
                if action == 1 and st.session_state.live_cash >= bet:
                    st.session_state.live_yes  += (bet * 0.998) / yp
                    st.session_state.live_cash -= bet
                elif action == 2 and st.session_state.live_yes > 0:
                    st.session_state.live_cash += st.session_state.live_yes * yp * 0.998
                    st.session_state.live_yes   = 0.0
                elif action == 3 and st.session_state.live_cash >= bet:
                    st.session_state.live_no   += (bet * 0.998) / np_
                    st.session_state.live_cash -= bet
                elif action == 4 and st.session_state.live_no > 0:
                    st.session_state.live_cash += st.session_state.live_no * np_ * 0.998
                    st.session_state.live_no    = 0.0

                st.session_state.live_step += 1
                t_new = st.session_state.live_step
                yp_new = float(np.clip(X[min(t_new, len(X)-1), YES_PROB_IDX if YES_PROB_IDX < X.shape[1] else 0], 0.01, 0.99))
                pv_new = (st.session_state.live_cash
                          + st.session_state.live_yes * yp_new
                          + st.session_state.live_no  * (1 - yp_new))
                st.session_state.live_portfolio.append(pv_new)
                st.session_state.live_actions.append(ACTION_LABELS[action])

        if step_btn:
            _do_step(1)
        if run10_btn:
            _do_step(10)

        # Display
        pv_now   = st.session_state.live_portfolio[-1]
        ret_now  = (pv_now / initial_cash - 1) * 100
        last_act = st.session_state.live_actions[-1] if st.session_state.live_actions else "—"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Portfolio", f"${pv_now:,.2f}", f"{ret_now:+.2f}%")
        m2.metric("Step", f"{st.session_state.live_step} / {len(X)}")
        m3.metric("Last Action", last_act)
        m4.metric("YES shares", f"{st.session_state.live_yes:.3f}")

        # Equity curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=st.session_state.live_portfolio, mode="lines",
            name="Portfolio", line=dict(color="#2563eb", width=2)
        ))
        fig.add_hline(y=initial_cash, line_dash="dash", line_color="grey",
                      annotation_text="Start")

        # Colour-code actions
        if st.session_state.live_actions:
            for i, act in enumerate(st.session_state.live_actions):
                if act != "HOLD":
                    fig.add_vline(
                        x=i+1,
                        line_width=1,
                        line_dash="dot",
                        line_color=ACTION_COLORS.get(act, "black"),
                    )

        fig.update_layout(
            title="Portfolio Equity Curve",
            xaxis_title="Step", yaxis_title="Value ($)",
            height=350, margin=dict(l=40, r=20, t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        # YES price overlay
        yes_prices = mkt_df["yes_prob"].values if "yes_prob" in mkt_df.columns else mkt_df["c"].values
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            y=yes_prices[:st.session_state.live_step+1],
            mode="lines", name="YES price",
            line=dict(color="#16a34a", width=1.5)
        ))
        fig2.update_layout(
            title="YES Implied Probability", height=200,
            xaxis_title="Step", yaxis_title="Prob",
            margin=dict(l=40, r=20, t=40, b=30),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — Backtest Results
# ══════════════════════════════════════════════════════════════════════════════
with tab_backtest:
    st.header("📈 Backtest Results")

    summary = load_summary(results_csv)
    if summary is None:
        st.info("Run `python training/evaluate.py` to generate results.")
    else:
        st.dataframe(
            summary.style.format(
                {"mean_roi": "{:+.2f}%", "mean_sharpe": "{:.3f}", "mean_maxdd": "{:+.2f}%"}
            ).background_gradient(subset=["mean_roi"], cmap="RdYlGn"),
            use_container_width=True,
        )

        # Render saved plots
        img_dir = ROOT / "results"
        for img_path in sorted(img_dir.glob("*.png")):
            st.subheader(img_path.stem.replace("_", " ").title())
            st.image(str(img_path))


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — Market Browser
# ══════════════════════════════════════════════════════════════════════════════
with tab_browser:
    st.header("📰 Market Browser")

    df_proc = load_processed(processed_path)
    if df_proc is None:
        st.warning("Processed features not found.")
    else:
        markets_meta = (
            df_proc.groupby("market_id")
            .agg(
                question =("question",  "first"),
                outcome  =("outcome",   "first"),
                category =("category",  "first") if "category" in df_proc.columns else ("outcome", "first"),
                n_candles=("t",         "count"),
                split    =("split",     "first"),
            )
            .reset_index()
        )
        markets_meta["outcome_label"] = markets_meta["outcome"].map({1: "✅ YES", 0: "❌ NO", -1: "❓ Unknown"})

        # Filters
        col_f1, col_f2 = st.columns(2)
        split_filter   = col_f1.multiselect("Split", ["train", "test"], default=["train", "test"])
        outcome_filter = col_f2.multiselect("Outcome", ["✅ YES", "❌ NO"], default=["✅ YES", "❌ NO"])

        filtered = markets_meta[
            markets_meta["split"].isin(split_filter) &
            markets_meta["outcome_label"].isin(outcome_filter)
        ].reset_index(drop=True)

        if filtered.empty:
            st.info("No markets match the selected filters.")
        else:
            filtered = add_question_labels(filtered)
            st.dataframe(
                filtered[["display_question", "outcome_label", "n_candles", "split"]].rename(
                    columns={
                        "display_question": "Question",
                        "outcome_label": "Outcome",
                        "n_candles": "# Candles",
                        "split": "Split",
                    }
                ),
                use_container_width=True,
            )

            # Price chart for selected market (question-first display)
            sel_label = st.selectbox(
                "Inspect market price history",
                filtered["display_question"].tolist(),
            )
            sel_market = filtered.loc[
                filtered["display_question"] == sel_label, "market_id"
            ].iloc[0]
            mkt_df = df_proc[df_proc["market_id"] == sel_market].sort_values("t")
            if not mkt_df.empty and "c" in mkt_df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=mkt_df["t"], y=mkt_df["c"],
                    mode="lines", name="YES price",
                    line=dict(color="#2563eb", width=1.5),
                ))
                if "volume" in mkt_df.columns:
                    fig.add_trace(go.Bar(
                        x=mkt_df["t"], y=mkt_df["volume"],
                        name="Volume", yaxis="y2",
                        marker_color="rgba(37,99,235,0.2)",
                    ))
                    fig.update_layout(
                        yaxis2=dict(overlaying="y", side="right", title="Volume"),
                    )
                outcome_val = int(mkt_df["outcome"].iloc[0])
                fig.update_layout(
                    title=f"{mkt_df['question'].iloc[0][:80]}…  → {'YES' if outcome_val == 1 else 'NO'}",
                    xaxis_title="Time", yaxis_title="YES Price ($)",
                    height=350, margin=dict(l=40, r=60, t=50, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)
