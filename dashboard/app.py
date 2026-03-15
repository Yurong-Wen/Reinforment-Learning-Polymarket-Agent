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

# Allow imports from project root regardless of working directory.
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
    """
    Try to load the PPO model from the configured path, then fall back through
    a chain of alternatives so the dashboard still works during and after training.
    Priority order: configured path → best checkpoint → latest periodic checkpoint.
    """
    p = Path(model_path)
    candidates = []

    # Primary target from config/sidebar.
    candidates.append(p)
    if p.suffix != ".zip":
        candidates.append(p.with_suffix(".zip"))

    # Fallback to the best model saved by EvalCallback when training finishes.
    candidates.append(ROOT / "models" / "best" / "best_model.zip")
    ckpts = sorted(
        (ROOT / "models" / "checkpoints").glob("ppo_poly_*_steps.zip"),
        key=lambda fp: fp.stat().st_mtime,
        reverse=True,
    )
    # Use the most recently written checkpoint if no final model exists yet.
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
    This deduplication is necessary because Polymarket sometimes lists the same
    question across multiple binary market contracts.
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
    # Use an emoji header instead of an external favicon URL to avoid
    # making an outbound HTTP request on every sidebar render.
    st.markdown("## 🤖 Polymarket RL")
    st.caption("Reinforcement Learning on Prediction Markets")
    st.divider()

    cfg         = load_config()
    model_path  = st.text_input("Model path", value=cfg["evaluation"]["model_path"])
    initial_cash = st.number_input(
        "Starting capital ($)",
        min_value=10.0,
        max_value=100_000.0,
        value=float(cfg["environment"]["initial_cash"]),
        step=100.0,
        help="Initial portfolio value used in the live simulation and evaluation runs.",
    )
    max_pos = st.slider(
        "Max position %",
        min_value=0.05,
        max_value=0.50,
        value=float(cfg["environment"]["max_position_pct"]),
        step=0.05,
        help="Maximum fraction of the current portfolio that can be wagered on a single action.",
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

    # Brief project description so anyone opening the dashboard understands context.
    with st.expander("ℹ️ About this project"):
        st.markdown(
            """
            This dashboard visualises a **Proximal Policy Optimisation (PPO)** agent
            trained to trade YES/NO shares on resolved [Polymarket](https://polymarket.com)
            prediction markets.

            The trading problem is formulated as a **Partially Observable Markov Decision
            Process (POMDP)**: the agent observes market prices, volumes, and technical
            indicators but never sees the true outcome probability.  The agent must learn
            from delayed resolution payoffs at the end of each market.

            **Pipeline:** fetch data → engineer features → train PPO (1 M steps) →
            backtest against 4 baselines → explore results here.
            """
        )

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

        col1.metric(
            "PPO Mean ROI",
            f"{ppo_roi:+.2f}%",
            f"vs best baseline {ppo_roi - base_roi:+.2f}%",
            help="Average percentage return of the PPO agent across 50 test episodes.",
        )
        col2.metric(
            "PPO Sharpe",
            f"{ppo_sharpe:.3f}",
            help="Risk-adjusted return (mean return / std dev, annualised).  Higher is better.",
        )
        col3.metric(
            "Best Baseline",
            f"{base_roi:+.2f}%",
            str(best_base["agent"]) if best_base is not None else "—",
            help="The best-performing non-PPO strategy from the four benchmarks.",
        )

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
            "No backtest results yet.  \n\n"
            "Run the following commands to generate them:\n"
            "```bash\n"
            "python run.py --train    # Train the PPO agent\n"
            "python run.py --evaluate # Run backtest and save results/summary.csv\n"
            "```"
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

    if model is None:
        st.info(
            "**No trained model found.**  "
            "The step-through will default to HOLD for every action.  \n\n"
            "To load a real model, run:\n"
            "```bash\n"
            "python run.py --train\n"
            "```"
        )

    if df_proc is None:
        st.warning(
            "Processed features not found.  "
            "Run `python data/preprocessing.py` first."
        )
    else:
        from data.preprocessing import FEATURE_COLS

        # Market selector (question-first display, id used internally).
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

        # Build feature matrix from only the columns that are present in this market's data.
        feat_cols_present = [c for c in FEATURE_COLS if c in mkt_df.columns]
        X = mkt_df[feat_cols_present].fillna(0).values.astype(np.float32)

        # Initialise session state for the episode whenever a new market is selected.
        if "live_step" not in st.session_state or st.session_state.get("live_market") != sel_market:
            st.session_state.live_step      = 0
            st.session_state.live_cash      = initial_cash
            st.session_state.live_yes       = 0.0
            st.session_state.live_no        = 0.0
            st.session_state.live_portfolio = [initial_cash]
            st.session_state.live_actions   = []
            st.session_state.live_market    = sel_market

        # Controls
        button_col_next, button_col_run10, button_col_runall, button_col_reset = st.columns([1, 1, 1, 2])
        step_btn   = button_col_next.button("▶ Next Step")
        run10_btn  = button_col_run10.button("⏩ Run 10 Steps")
        runall_btn = button_col_runall.button("⏭ Run to End")
        reset_btn  = button_col_reset.button("🔄 Reset Episode")

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

        def _do_step(n_steps: int = 1):
            """
            Advance the simulated episode by n_steps ticks.
            This function uses a simplified portfolio model that mirrors the
            PolymarketEnv logic directly in session state, without inverting the
            StandardScaler — the observation fed to the PPO model therefore
            uses pre-scaled feature values as they appear in the processed parquet.
            """
            from env.polymarket_env import PolymarketEnv, YES_PROB_IDX
            for _ in range(n_steps):
                current_step_index = st.session_state.live_step
                if current_step_index >= len(X) - 1:
                    break
                yes_price = float(np.clip(X[current_step_index, YES_PROB_IDX if YES_PROB_IDX < X.shape[1] else 0], 0.01, 0.99))
                no_price  = 1.0 - yes_price
                portfolio_value = (
                    st.session_state.live_cash
                    + st.session_state.live_yes * yes_price
                    + st.session_state.live_no  * no_price
                )

                # Build the full observation vector by concatenating market features
                # with the four normalised portfolio state features.
                portfolio_feats = np.array([
                    st.session_state.live_yes  * yes_price / (portfolio_value + 1e-8),
                    st.session_state.live_no   * no_price  / (portfolio_value + 1e-8),
                    st.session_state.live_cash             / (portfolio_value + 1e-8),
                    1.0 - current_step_index / (len(X) - 1 + 1e-8),
                ], dtype=np.float32)
                obs = np.concatenate([X[current_step_index], portfolio_feats])

                if model is not None:
                    action, _ = model.predict(obs, deterministic=True)
                    action    = int(action)
                else:
                    action = 0  # Default to HOLD when no model is loaded.

                # Apply the chosen action using the same trade mechanics as PolymarketEnv.
                bet_amount_dollars = max_pos * portfolio_value
                if action == 1 and st.session_state.live_cash >= bet_amount_dollars:
                    st.session_state.live_yes  += (bet_amount_dollars * 0.998) / yes_price
                    st.session_state.live_cash -= bet_amount_dollars
                elif action == 2 and st.session_state.live_yes > 0:
                    st.session_state.live_cash += st.session_state.live_yes * yes_price * 0.998
                    st.session_state.live_yes   = 0.0
                elif action == 3 and st.session_state.live_cash >= bet_amount_dollars:
                    st.session_state.live_no   += (bet_amount_dollars * 0.998) / no_price
                    st.session_state.live_cash -= bet_amount_dollars
                elif action == 4 and st.session_state.live_no > 0:
                    st.session_state.live_cash += st.session_state.live_no * no_price * 0.998
                    st.session_state.live_no    = 0.0

                st.session_state.live_step += 1
                updated_yes_price = float(np.clip(
                    X[min(st.session_state.live_step, len(X) - 1), YES_PROB_IDX if YES_PROB_IDX < X.shape[1] else 0],
                    0.01, 0.99,
                ))
                updated_portfolio_value = (
                    st.session_state.live_cash
                    + st.session_state.live_yes * updated_yes_price
                    + st.session_state.live_no  * (1 - updated_yes_price)
                )
                st.session_state.live_portfolio.append(updated_portfolio_value)
                st.session_state.live_actions.append(ACTION_LABELS[action])

        if step_btn:
            _do_step(1)
        if run10_btn:
            _do_step(10)
        if runall_btn:
            # Run all remaining steps to the end of the episode.
            _do_step(len(X))

        # ── Episode progress bar ──────────────────────────────────────────
        episode_completion_fraction = min(st.session_state.live_step / max(len(X) - 1, 1), 1.0)
        st.progress(episode_completion_fraction, text=f"Episode progress: {st.session_state.live_step}/{len(X)} steps")

        # ── KPI metrics ───────────────────────────────────────────────────
        current_portfolio_value = st.session_state.live_portfolio[-1]
        current_return_percentage = (current_portfolio_value / initial_cash - 1) * 100
        last_action_label = st.session_state.live_actions[-1] if st.session_state.live_actions else "—"

        metric_col_portfolio, metric_col_step, metric_col_action, metric_col_yes_shares, metric_col_no_shares = st.columns(5)
        metric_col_portfolio.metric(
            "Portfolio",
            f"${current_portfolio_value:,.2f}",
            f"{current_return_percentage:+.2f}%",
            help="Current mark-to-market portfolio value.",
        )
        metric_col_step.metric(
            "Step",
            f"{st.session_state.live_step} / {len(X)}",
            help="Current candle index out of total candles for this market.",
        )
        metric_col_action.metric(
            "Last Action",
            last_action_label,
            help="The action the PPO agent took on the most recent step.",
        )
        metric_col_yes_shares.metric(
            "YES shares",
            f"{st.session_state.live_yes:.3f}",
            help="Number of YES-outcome shares currently held.",
        )
        # NO shares were previously missing from the display — added to show the
        # complete position since the agent can hold both sides simultaneously.
        metric_col_no_shares.metric(
            "NO shares",
            f"{st.session_state.live_no:.3f}",
            help="Number of NO-outcome shares currently held.",
        )

        # ── Equity curve ──────────────────────────────────────────────────
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=st.session_state.live_portfolio, mode="lines",
            name="Portfolio", line=dict(color="#2563eb", width=2)
        ))
        fig.add_hline(y=initial_cash, line_dash="dash", line_color="grey",
                      annotation_text="Start")

        # Add a vertical marker at each non-HOLD action to highlight trading events.
        if st.session_state.live_actions:
            for i, act in enumerate(st.session_state.live_actions):
                if act != "HOLD":
                    fig.add_vline(
                        x=i + 1,
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

        # ── YES price overlay ─────────────────────────────────────────────
        yes_prices = mkt_df["yes_prob"].values if "yes_prob" in mkt_df.columns else mkt_df["c"].values
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            y=yes_prices[:st.session_state.live_step + 1],
            mode="lines", name="YES price",
            line=dict(color="#16a34a", width=1.5)
        ))
        fig2.update_layout(
            title="YES Implied Probability", height=200,
            xaxis_title="Step", yaxis_title="Prob",
            margin=dict(l=40, r=20, t=40, b=30),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── Action history table ──────────────────────────────────────────
        if st.session_state.live_actions:
            with st.expander("📋 Action History (last 10 steps)"):
                recent_actions = st.session_state.live_actions[-10:]
                action_history_df = pd.DataFrame({
                    "Step":   list(range(
                        max(1, st.session_state.live_step - len(recent_actions) + 1),
                        st.session_state.live_step + 1,
                    )),
                    "Action": recent_actions,
                })
                st.dataframe(action_history_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — Backtest Results
# ══════════════════════════════════════════════════════════════════════════════
with tab_backtest:
    st.header("📈 Backtest Results")

    # Metric glossary so readers understand what each column means.
    with st.expander("📖 Metric Definitions"):
        st.markdown(
            """
            | Metric | Meaning |
            |--------|---------|
            | **Mean ROI (%)** | Average percentage return across 50 test episodes.  Positive = profitable. |
            | **Mean Sharpe** | Risk-adjusted return (annualised mean / std dev).  Higher is better; >1.0 is considered good. |
            | **Mean MaxDD (%)** | Average worst peak-to-trough loss per episode.  Less negative = smaller drawdown risk. |
            """
        )

    summary = load_summary(results_csv)
    if summary is None:
        st.info(
            "No backtest results found.  \n\n"
            "Run:\n"
            "```bash\n"
            "python run.py --evaluate\n"
            "```"
        )
    else:
        st.dataframe(
            summary.style.format(
                {"mean_roi": "{:+.2f}%", "mean_sharpe": "{:.3f}", "mean_maxdd": "{:+.2f}%"}
            ).background_gradient(subset=["mean_roi"], cmap="RdYlGn"),
            use_container_width=True,
        )

        # Render saved static plots generated by evaluate.py.
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
        st.warning("Processed features not found.  Run `python data/preprocessing.py` first.")
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
        filter_col_split, filter_col_outcome = st.columns(2)
        split_filter   = filter_col_split.multiselect("Split", ["train", "test"], default=["train", "test"])
        outcome_filter = filter_col_outcome.multiselect("Outcome", ["✅ YES", "❌ NO"], default=["✅ YES", "❌ NO"])

        # Keyword search lets users quickly find markets by topic.
        keyword_search = st.text_input("🔍 Search markets", "", placeholder="e.g. election, bitcoin, sports...")

        filtered = markets_meta[
            markets_meta["split"].isin(split_filter) &
            markets_meta["outcome_label"].isin(outcome_filter)
        ].reset_index(drop=True)

        # Apply keyword filter when the user types something into the search box.
        if keyword_search.strip():
            filtered = filtered[
                filtered["question"].str.contains(keyword_search.strip(), case=False, na=False)
            ].reset_index(drop=True)

        # Category filter is only shown when the dataset includes non-empty category values.
        if "category" in filtered.columns and filtered["category"].notna().any():
            all_categories = sorted(filtered["category"].dropna().unique().tolist())
            if all_categories:
                selected_categories = st.multiselect(
                    "Category",
                    options=all_categories,
                    default=all_categories,
                )
                filtered = filtered[filtered["category"].isin(selected_categories)].reset_index(drop=True)

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

            # Price chart for selected market (question-first display).
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
                # Use standard ASCII ellipsis instead of the Unicode character to
                # avoid rendering issues on some operating systems and fonts.
                fig.update_layout(
                    title=f"{mkt_df['question'].iloc[0][:80]}...  -> {'YES' if outcome_val == 1 else 'NO'}",
                    xaxis_title="Time", yaxis_title="YES Price ($)",
                    height=350, margin=dict(l=40, r=60, t=50, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)
