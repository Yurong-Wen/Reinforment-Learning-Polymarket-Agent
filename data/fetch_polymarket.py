"""
fetch_polymarket.py
───────────────────
Collects resolved binary market data from Polymarket's public APIs:
  • Gamma API  – market metadata & resolution outcomes
  • CLOB API   – per-market price history and trade records

Run standalone:
    python data/fetch_polymarket.py
"""

import os
import time
import logging
from typing import Optional

import requests
import pandas as pd
from tqdm import tqdm
from rich.console import Console
from rich.logging import RichHandler

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("fetch_polymarket")
console = Console()

# ── API Endpoints ─────────────────────────────────────────────────────────────
GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE  = "https://clob.polymarket.com"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get(url: str, params: dict = None, retries: int = 3, backoff: float = 2.0):
    """GET with automatic retry + exponential back-off."""
    for attempt in range(retries):
        try:
            resp = requests.get(
                url,
                params=params,
                timeout=15,
                headers={"User-Agent": "polymarket-rl/1.0"},
            )
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            # Do not retry on non-rate-limit client errors (e.g. bad query params).
            if status is not None and 400 <= status < 500 and status != 429:
                raise
            if attempt == retries - 1:
                raise
            wait = backoff ** attempt
            log.warning(f"Request failed ({exc}). Retrying in {wait:.1f}s …")
            time.sleep(wait)
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            wait = backoff ** attempt
            log.warning(f"Request failed ({exc}). Retrying in {wait:.1f}s …")
            time.sleep(wait)


# ── Gamma API ─────────────────────────────────────────────────────────────────

def fetch_resolved_markets(limit: int = 100, offset: int = 0) -> list[dict]:
    """Return a page of resolved markets from the Gamma API."""
    data = _get(
        f"{GAMMA_BASE}/markets",
        params={
            "closed": "true",
            "limit":  limit,
            "offset": offset,
            "order":  "volume",
            "ascending": "false",
        },
    )
    # API returns a list directly
    return data if isinstance(data, list) else data.get("markets", [])


def collect_market_metadata(n_markets: int = 100) -> list[dict]:
    """Fetch metadata for `n_markets` high-volume resolved markets."""
    markets, offset = [], 0
    with console.status("[bold green]Fetching market metadata …"):
        while len(markets) < n_markets:
            batch = fetch_resolved_markets(limit=100, offset=offset)
            if not batch:
                log.warning("No more markets returned by Gamma API.")
                break
            markets.extend(batch)
            offset += 100
            time.sleep(0.4)
    log.info(f"Collected metadata for {len(markets[:n_markets])} markets.")
    return markets[:n_markets]


# ── CLOB API ──────────────────────────────────────────────────────────────────

def _to_unix_seconds(ts: Optional[str]) -> Optional[int]:
    """Convert ISO/date-like timestamp into unix seconds."""
    if not ts:
        return None
    dt = pd.to_datetime(ts, utc=True, errors="coerce")
    if pd.isna(dt):
        return None
    return int(dt.timestamp())


def fetch_price_history(
    token_id: str,
    interval: str = "1h",
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV-style price history for a single market token.

    Tries several parameter combinations so that minor API changes don't
    silently return empty data.

    Parameters
    ----------
    token_id : str
        The token ID for the YES share of the market.
    interval : str
        Candle interval – e.g. "1h", "6h", "1d".

    Returns
    -------
    pd.DataFrame with columns: t, o, h, l, c, volume
    """
    # Fidelity values to try (minutes per bucket → maps to interval)
    INTERVAL_FIDELITY = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "6h": 360, "1d": 1440}
    fidelity = INTERVAL_FIDELITY.get(interval, 60)

    history = []

    # If we have a known market time range, fetch in chunks to avoid API limits.
    if start_ts is not None and end_ts is not None and end_ts > start_ts:
        chunk_seconds = 21 * 24 * 3600  # conservative to avoid "interval too long"
        t0 = start_ts
        while t0 < end_ts:
            t1 = min(t0 + chunk_seconds, end_ts)
            try:
                data = _get(
                    f"{CLOB_BASE}/prices-history",
                    params={
                        "market": token_id,
                        "interval": interval,
                        "fidelity": fidelity,
                        "startTs": t0,
                        "endTs": t1,
                    },
                )
                chunk = data.get("history", []) if isinstance(data, dict) else []
                if chunk:
                    history.extend(chunk)
            except Exception:
                pass
            t0 = t1
    else:
        # Fallback for unknown time ranges.
        param_variants = [
            {"market": token_id, "interval": interval, "fidelity": fidelity},
            {"market": token_id, "interval": interval},
        ]
        for params in param_variants:
            try:
                data = _get(f"{CLOB_BASE}/prices-history", params=params)
                history = data.get("history", []) if isinstance(data, dict) else []
                if history:
                    break
            except Exception:
                continue

    if not history:
        return pd.DataFrame()

    df = pd.DataFrame(history)
    # Normalise column names
    df.columns = [c.lower() for c in df.columns]
    if "t" not in df.columns and "time" in df.columns:
        df = df.rename(columns={"time": "t"})
    elif "t" not in df.columns and "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "t"})

    if "t" not in df.columns:
        return pd.DataFrame()

    df["t"] = pd.to_datetime(df["t"], unit="s", utc=True, errors="coerce")
    df = df.dropna(subset=["t"])
    df = df.drop_duplicates(subset=["t"], keep="last")

    # Ensure numeric OHLCV – the API sometimes returns "p" (price) not "c" (close)
    if "c" not in df.columns and "p" in df.columns:
        df["c"] = pd.to_numeric(df["p"], errors="coerce")
    for col in ["o", "h", "l", "c"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            # Synthesise missing OHLC from close price
            if "c" in df.columns:
                df[col] = df["c"]

    if "volume" not in df.columns:
        df["volume"] = 0.0
    else:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    df = df.dropna(subset=["c"])
    return df.sort_values("t").reset_index(drop=True)


def fetch_market_trades(market_id: str, limit: int = 500) -> pd.DataFrame:
    """Fetch the most recent trade events for volume/liquidity analysis."""
    try:
        data = _get(
            f"{CLOB_BASE}/trades",
            params={"market": market_id, "limit": limit},
        )
        records = data.get("data", data) if isinstance(data, dict) else data
        return pd.DataFrame(records)
    except Exception as exc:
        log.debug(f"Could not fetch trades for {market_id}: {exc}")
        return pd.DataFrame()


# ── Main Collection Pipeline ──────────────────────────────────────────────────

def _extract_token_id(market: dict) -> Optional[str]:
    """
    Extract the YES token ID from a Gamma API market dict.

    Polymarket stores token IDs in several possible locations:
      1. market["clobTokenIds"]   – JSON string: '["yes_id", "no_id"]'  (most common)
      2. market["tokens"]         – list of dicts with token_id + outcome
      3. market["outcomes"]       – similar list format
    """
    import json as _json

    # ── 1. clobTokenIds (primary field in Gamma API) ──────────────────────
    clob_raw = market.get("clobTokenIds")
    if clob_raw:
        try:
            ids = _json.loads(clob_raw) if isinstance(clob_raw, str) else clob_raw
            if isinstance(ids, list) and len(ids) >= 1:
                yes_id = ids[0]            # index 0 = YES, index 1 = NO
                if yes_id:
                    return str(yes_id)
        except (ValueError, TypeError):
            pass

    # ── 2. tokens list ────────────────────────────────────────────────────
    tokens = market.get("tokens") or []
    if isinstance(tokens, str):
        try:
            tokens = _json.loads(tokens)
        except (ValueError, TypeError):
            tokens = []
    for tok in tokens:
        if isinstance(tok, dict):
            outcome = str(tok.get("outcome", "")).lower()
            if outcome in ("yes", "true", "1"):
                tid = tok.get("token_id") or tok.get("tokenId") or tok.get("id")
                if tid:
                    return str(tid)

    # ── 3. outcomes list ──────────────────────────────────────────────────
    outcomes_list = market.get("outcomes") or []
    if isinstance(outcomes_list, str):
        try:
            outcomes_list = _json.loads(outcomes_list)
        except (ValueError, TypeError):
            outcomes_list = []
    for item in outcomes_list:
        if isinstance(item, dict):
            outcome = str(item.get("outcome", item.get("name", ""))).lower()
            if outcome in ("yes", "true"):
                tid = item.get("token_id") or item.get("tokenId")
                if tid:
                    return str(tid)

    return None


def _extract_outcome(market: dict) -> int:
    """Return 1 if Yes resolved, 0 if No resolved, -1 if unknown."""
    import json as _json

    resolution = str(market.get("resolutionValue", "")).lower()
    if resolution in ("1", "yes", "true"):
        return 1
    if resolution in ("0", "no", "false"):
        return 0

    # Newer Gamma responses often expose only final outcomePrices, e.g. ["1","0"].
    outcome_prices = market.get("outcomePrices")
    if outcome_prices is not None:
        try:
            vals = _json.loads(outcome_prices) if isinstance(outcome_prices, str) else outcome_prices
            if isinstance(vals, list) and len(vals) >= 2:
                p0 = float(vals[0])
                p1 = float(vals[1])
                if p0 > p1:
                    return 1
                if p1 > p0:
                    return 0
        except (ValueError, TypeError):
            pass

    return -1


def debug_market_structure(n_samples: int = 3):
    """
    Print the raw structure of the first `n_samples` markets so you can
    inspect which fields the Gamma API is actually returning.
    Run this if data collection fails:
        python -c "from data.fetch_polymarket import debug_market_structure; debug_market_structure()"
    """
    import json as _json
    markets = fetch_resolved_markets(limit=n_samples, offset=0)
    for i, m in enumerate(markets[:n_samples]):
        print(f"\n{'='*60}")
        print(f"Market {i+1}: {m.get('question', '')[:80]}")
        print(f"{'='*60}")
        # Show keys + truncated values
        for k, v in m.items():
            val_str = str(v)[:120]
            print(f"  {k:<30} {val_str}")


def collect_dataset(
    n_markets:    int = 100,
    price_interval: str = "1h",
    save_path:    str = "data/raw/polymarket_raw.parquet",
    min_rows:     int = 5,
) -> pd.DataFrame:
    """
    Full pipeline: collect metadata + price history → save parquet.

    Parameters
    ----------
    n_markets : int
        How many resolved markets to target.
    price_interval : str
        OHLCV candle size.
    save_path : str
        Output parquet path.
    min_rows : int
        Skip markets with fewer than this many price candles.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    markets = collect_market_metadata(n_markets)

    all_dfs = []
    skipped = 0

    no_token_count  = 0
    too_short_count = 0
    error_count     = 0

    for mkt in tqdm(markets, desc="Downloading price histories"):
        token_id = _extract_token_id(mkt)
        if not token_id:
            no_token_count += 1
            skipped += 1
            log.debug(f"No token_id for market: {mkt.get('question','')[:60]}")
            continue

        try:
            start_ts = (
                _to_unix_seconds(mkt.get("startDate"))
                or _to_unix_seconds(mkt.get("acceptingOrdersTimestamp"))
                or _to_unix_seconds(mkt.get("createdAt"))
            )
            end_ts = (
                _to_unix_seconds(mkt.get("closedTime"))
                or _to_unix_seconds(mkt.get("endDate"))
                or _to_unix_seconds(mkt.get("umaEndDate"))
            )
            if start_ts is not None:
                start_ts -= 6 * 3600  # include pre-open activity buffer
            if end_ts is not None:
                end_ts += 6 * 3600    # include late settlement prints

            prices = fetch_price_history(
                token_id,
                interval=price_interval,
                start_ts=start_ts,
                end_ts=end_ts,
            )
            if prices.empty or len(prices) < min_rows:
                too_short_count += 1
                skipped += 1
                log.debug(f"Too short ({len(prices)} rows) for token {token_id[:16]}…")
                continue

            prices["market_id"] = token_id
            prices["question"]  = mkt.get("question", "")
            prices["outcome"]   = _extract_outcome(mkt)
            prices["category"]  = mkt.get("category", "")
            all_dfs.append(prices)

        except Exception as exc:
            log.warning(f"Skipping market {token_id}: {exc}")
            error_count += 1
            skipped += 1

        time.sleep(0.30)  # Respect API rate limits

    log.info(
        f"Skip breakdown — no token_id: {no_token_count}  |  "
        f"too short: {too_short_count}  |  errors: {error_count}"
    )

    if not all_dfs:
        raise RuntimeError(
            "No market data was collected.\n\n"
            "Possible causes:\n"
            "  1. The Gamma API token ID field name changed.\n"
            "     → Run the debug helper to inspect the raw market structure:\n"
            "       python -c \"from data.fetch_polymarket import debug_market_structure; "
            "debug_market_structure()\"\n\n"
            "  2. The CLOB prices-history API returned empty responses.\n"
            "     → Try a longer interval: set price_interval='1d' in config.yaml\n\n"
            "  3. All markets had fewer than min_rows candles.\n"
            "     → min_rows is already 5; this should not be the issue."
        )

    df = pd.concat(all_dfs, ignore_index=True)
    df.to_parquet(save_path, index=False)

    console.print(
        f"\n[bold green]✓ Saved[/] {len(df):,} rows across "
        f"[cyan]{df['market_id'].nunique()}[/] markets → [yellow]{save_path}[/]\n"
        f"  Skipped: {skipped} markets (too few rows or missing token ID)"
    )
    return df


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml

    with open("configs/config.yaml") as fh:
        cfg = yaml.safe_load(fh)

    collect_dataset(
        n_markets=cfg["data"]["n_markets"],
        price_interval=cfg["data"]["price_interval"],
        save_path=cfg["data"]["raw_path"],
    )
