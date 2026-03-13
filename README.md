# Polymarket RL

End-to-end reinforcement learning pipeline for prediction-market trading on historical Polymarket data.

It includes:
- data collection from Polymarket public APIs
- optional news sentiment scoring (FinBERT + NewsAPI)
- feature engineering and train/test split
- PPO training with Stable-Baselines3
- evaluation against baseline agents
- Streamlit dashboard for exploration and demos

## Project Structure

```text
Poly/
├── configs/
│   └── config.yaml
├── data/
│   ├── fetch_polymarket.py
│   ├── fetch_sentiment.py
│   └── preprocessing.py
├── env/
│   └── polymarket_env.py
├── agents/
│   └── baselines.py
├── training/
│   ├── train.py
│   └── evaluate.py
├── dashboard/
│   └── app.py
├── run.py
└── requirements.txt
```

## Requirements

- Python 3.10+ (3.12 works)
- macOS/Linux/WSL recommended
- internet access for Polymarket APIs
- optional `NEWSAPI_KEY` for non-zero sentiment

## Setup

1. Create and activate a virtual environment.
2. Install dependencies.
3. (Optional) add NewsAPI key.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` only if you want real sentiment values:

```bash
echo "NEWSAPI_KEY=your_key_here" > .env
```

Without `NEWSAPI_KEY`, the pipeline still runs and uses `0.0` sentiment.

## Configuration

Main configuration is in `configs/config.yaml`.

Key fields:
- `data.n_markets`: number of resolved markets to collect
- `data.price_interval`: price candle interval (`1h`, `6h`, `1d`, ...)
- `ppo.total_timesteps`: training length
- `evaluation.model_path`: path prefix where model is saved
- `dashboard.port`: Streamlit port

## Running the Pipeline

### Full pipeline

```bash
python run.py --all
```

Runs:
1. fetch data
2. sentiment scoring
3. preprocessing
4. PPO training
5. evaluation
6. dashboard

### Stage-by-stage

```bash
python run.py --fetch
python run.py --sentiment
python run.py --preprocess
python run.py --train
python run.py --evaluate
python run.py --dashboard
```

### Common combinations

Train + evaluate without launching dashboard:

```bash
python run.py --fetch --sentiment --preprocess --train --evaluate
```

Run dashboard only:

```bash
streamlit run dashboard/app.py
```

## Outputs

### Data
- `data/raw/polymarket_raw.parquet`
- `data/sentiment/sentiment_scores.parquet`
- `data/processed/features.parquet`

### Models and training logs
- `models/ppo_polymarket_final.zip`
- `models/ppo_polymarket_final_scaler.pkl`
- `models/best/` (best checkpoints from eval callback)
- `models/checkpoints/` (periodic checkpoints)
- `runs/` (TensorBoard logs)

### Evaluation artifacts
- `results/summary.csv`
- `results/backtest_comparison.png`
- `results/drawdown_comparison.png`
- `results/return_distribution.png`
- `results/action_distribution.png`

## Monitoring Training

TensorBoard:

```bash
tensorboard --logdir runs/
```

Then open the local URL printed in terminal.

## Dashboard

The Streamlit dashboard includes:
- Overview KPIs and result charts
- PPO step-through simulation on a selected market
- Backtest results table and plots
- Market browser with filters and price/volume chart

Launch:

```bash
python run.py --dashboard
```

or:

```bash
streamlit run dashboard/app.py
```

## Troubleshooting

- **No market data collected**
  - check internet connectivity
  - reduce `data.n_markets` temporarily (for faster debugging)
  - try `data.price_interval: "1d"` in `configs/config.yaml`

- **Sentiment warning about missing key**
  - set `NEWSAPI_KEY` in `.env` to enable real sentiment
  - otherwise warning is expected, and score defaults to zero

- **Long training times**
  - reduce `ppo.total_timesteps`
  - reduce `ppo.n_envs` if machine resources are limited

- **Matplotlib/font cache warnings on macOS**
  - usually harmless for headless runs
  - optionally set `MPLCONFIGDIR` to a writable directory

## Reproducibility Notes

- Train/test split is market-level (not row-level) to reduce leakage.
- Preprocessing and training regenerate processed features from raw inputs.
- Evaluation compares PPO against:
  - `AlwaysBuyYes`
  - `AlwaysBuyNo`
  - `MarketOdds`
  - `Random`
