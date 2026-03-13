"""
fetch_sentiment.py
──────────────────
Downloads news headlines for each market question and scores them with
FinBERT (finance-tuned BERT).  Produces a parquet of per-market daily
sentiment scores that the preprocessing pipeline merges into the feature
matrix.

Requirements
------------
  pip install transformers newsapi-python torch

Usage
-----
  # Set your key in .env:  NEWSAPI_KEY=<your_key>
  python data/fetch_sentiment.py
"""

import os
import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from rich.logging import RichHandler

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("fetch_sentiment")


# ── FinBERT Sentiment ─────────────────────────────────────────────────────────

_pipeline = None  # Lazy-load to avoid slow imports at module level

def _get_pipeline():
    """Lazy-initialise the FinBERT pipeline (loaded once per session)."""
    global _pipeline
    if _pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline
            log.info("Loading FinBERT model (first run may download ~500 MB) …")
            _pipeline = hf_pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                truncation=True,
                max_length=512,
            )
            log.info("FinBERT loaded ✓")
        except Exception as exc:
            log.error(f"Could not load FinBERT: {exc}. Sentiment will be 0.")
            _pipeline = None
    return _pipeline


def score_text(text: str) -> float:
    """
    Score a piece of text with FinBERT.

    Returns
    -------
    float in [-1, +1]
        +1 = strongly positive, −1 = strongly negative.
        Returns 0.0 if the model is unavailable.
    """
    if not text or not text.strip():
        return 0.0
    nlp = _get_pipeline()
    if nlp is None:
        return 0.0
    try:
        result = nlp(text[:512])[0]
        score  = float(result["score"])
        label  = result["label"].lower()
        if label == "positive":
            return score
        elif label == "negative":
            return -score
        else:
            return 0.0  # 'neutral'
    except Exception as exc:
        log.debug(f"Scoring error: {exc}")
        return 0.0


# ── NewsAPI ───────────────────────────────────────────────────────────────────

def fetch_articles(
    query: str,
    from_date: str,
    to_date:   str,
    api_key:   str,
    page_size: int = 20,
) -> list[dict]:
    """Fetch news articles from NewsAPI."""
    try:
        from newsapi import NewsApiClient
        client = NewsApiClient(api_key=api_key)
        resp = client.get_everything(
            q=query,
            from_param=from_date,
            to=to_date,
            language="en",
            sort_by="publishedAt",
            page_size=page_size,
        )
        return resp.get("articles", [])
    except Exception as exc:
        log.warning(f"NewsAPI error for '{query}': {exc}")
        return []


def articles_to_score(articles: list[dict]) -> float:
    """Average FinBERT score over a list of article dicts."""
    scores = []
    for art in articles:
        headline = art.get("title", "") or ""
        desc     = art.get("description", "") or ""
        combined = f"{headline}. {desc}".strip()
        scores.append(score_text(combined))
    return float(np.mean(scores)) if scores else 0.0


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def compute_market_sentiment(
    raw_parquet:  str = "data/raw/polymarket_raw.parquet",
    save_path:    str = "data/sentiment/sentiment_scores.parquet",
    lookback_days: int = 7,
    newsapi_key:  Optional[str] = None,
) -> pd.DataFrame:
    """
    For each unique market in the raw dataset, query NewsAPI for headlines
    around the market's most active trading period and return a FinBERT score.

    Parameters
    ----------
    raw_parquet   : path to raw Polymarket parquet
    save_path     : output path
    lookback_days : how many days before market end to look for news
    newsapi_key   : override env-var NEWSAPI_KEY

    Returns
    -------
    DataFrame with columns: market_id, sentiment_score
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    key = newsapi_key or os.getenv("NEWSAPI_KEY", "")
    if not key:
        log.warning(
            "NEWSAPI_KEY not set – writing zero sentiment scores. "
            "Add NEWSAPI_KEY=<key> to your .env file to enable real sentiment."
        )

    df = pd.read_parquet(raw_parquet)
    markets = df.groupby("market_id").agg(
        question=("question", "first"),
        last_ts  =("t", "max"),
    ).reset_index()

    records = []
    for _, row in tqdm(markets.iterrows(), total=len(markets), desc="Scoring sentiment"):
        market_id = row["market_id"]
        question  = str(row["question"])
        last_ts   = pd.to_datetime(row["last_ts"])

        to_date   = last_ts.strftime("%Y-%m-%d")
        from_date = (last_ts - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        # Shorten question to a NewsAPI-friendly query (max 500 chars)
        query = question[:120]

        if key:
            articles = fetch_articles(query, from_date, to_date, key)
            score    = articles_to_score(articles)
            time.sleep(0.3)
        else:
            score = 0.0

        records.append({"market_id": market_id, "sentiment_score": score})

    result = pd.DataFrame(records)
    result.to_parquet(save_path, index=False)
    log.info(f"Saved sentiment scores for {len(result)} markets → {save_path}")
    return result


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml

    with open("configs/config.yaml") as fh:
        cfg = yaml.safe_load(fh)

    compute_market_sentiment(
        raw_parquet=cfg["data"]["raw_path"],
        save_path=cfg["data"]["sentiment_path"],
    )
