"""
run.py  —  Polymarket RL master runner
───────────────────────────────────────
Executes the full pipeline in order, or individual stages via CLI flags.

Usage examples
--------------
  # Full pipeline (data → preprocess → train → evaluate)
  python run.py --all

  # Individual stages
  python run.py --fetch          # 1. download Polymarket data
  python run.py --sentiment      # 2. compute news sentiment
  python run.py --preprocess     # 3. feature engineering
  python run.py --train          # 4. PPO training
  python run.py --evaluate       # 5. backtesting & plots
  python run.py --dashboard      # 6. launch Streamlit dashboard
"""

import argparse
import logging
import os
import subprocess
import sys

import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.rule import Rule

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log     = logging.getLogger("run")
console = Console()

CONFIG_PATH = "configs/config.yaml"


def load_config():
    with open(CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


# ── Stage functions ───────────────────────────────────────────────────────────

def stage_fetch(cfg):
    console.print(Rule("[bold blue]Stage 1 · Fetch Polymarket Data"))
    from data.fetch_polymarket import collect_dataset
    collect_dataset(
        n_markets=cfg["data"]["n_markets"],
        price_interval=cfg["data"]["price_interval"],
        save_path=cfg["data"]["raw_path"],
    )


def stage_sentiment(cfg):
    console.print(Rule("[bold blue]Stage 2 · Sentiment Scoring"))
    from data.fetch_sentiment import compute_market_sentiment
    compute_market_sentiment(
        raw_parquet=cfg["data"]["raw_path"],
        save_path=cfg["data"]["sentiment_path"],
    )


def stage_preprocess(cfg):
    console.print(Rule("[bold blue]Stage 3 · Feature Engineering & Preprocessing"))
    from data.preprocessing import build_dataset
    build_dataset(
        raw_parquet=cfg["data"]["raw_path"],
        sentiment_path=cfg["data"]["sentiment_path"],
        processed_path=cfg["data"]["processed_path"],
        train_ratio=cfg["data"]["train_ratio"],
        recent_weight=cfg["data"]["temporal_weights"]["recent"],
        old_weight=cfg["data"]["temporal_weights"]["old"],
    )


def stage_train(_cfg):
    console.print(Rule("[bold blue]Stage 4 · PPO Training"))
    from training.train import train
    train(config_path=CONFIG_PATH)


def stage_evaluate(_cfg):
    console.print(Rule("[bold blue]Stage 5 · Evaluation & Backtest Plots"))
    from training.evaluate import evaluate
    evaluate(config_path=CONFIG_PATH)


def stage_dashboard(_cfg):
    console.print(Rule("[bold blue]Stage 6 · Launching Streamlit Dashboard"))
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
         "--server.port", str(_cfg["dashboard"]["port"])],
        check=True,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Polymarket RL — full pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--all",        action="store_true", help="Run all stages")
    parser.add_argument("--fetch",      action="store_true", help="Fetch raw market data")
    parser.add_argument("--sentiment",  action="store_true", help="Compute news sentiment")
    parser.add_argument("--preprocess", action="store_true", help="Feature engineering")
    parser.add_argument("--train",      action="store_true", help="Train PPO agent")
    parser.add_argument("--evaluate",   action="store_true", help="Backtest & plots")
    parser.add_argument("--dashboard",  action="store_true", help="Launch dashboard")
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return

    console.print(Panel.fit(
        "[bold]Polymarket RL Pipeline[/]\n"
        "Team: Fatima PAGKALIWANGAN · Jose ROMERO MOLINA · Yurong WEN",
        border_style="blue",
    ))

    cfg = load_config()

    run_all = args.all
    if run_all or args.fetch:      stage_fetch(cfg)
    if run_all or args.sentiment:  stage_sentiment(cfg)
    if run_all or args.preprocess: stage_preprocess(cfg)
    if run_all or args.train:      stage_train(cfg)
    if run_all or args.evaluate:   stage_evaluate(cfg)
    if run_all or args.dashboard:  stage_dashboard(cfg)

    if run_all:
        console.print(Panel.fit(
            "[bold green]✓ Full pipeline complete![/]\n"
            "  • Data     → data/raw/ & data/processed/\n"
            "  • Model    → models/ppo_polymarket_final.zip\n"
            "  • Results  → results/\n"
            "  • Dashboard: [cyan]streamlit run dashboard/app.py[/]",
            border_style="green",
        ))


if __name__ == "__main__":
    main()
