from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


BANKS = {
    "RY.TO": "Royal Bank (RBC)",
    "TD.TO": "TD Bank",
    "BNS.TO": "Scotiabank",
    "BMO.TO": "BMO",
    "CM.TO": "CIBC",
}

_PROFILE = {
    "RY.TO": (0.09, 0.20, 0.038),
    "TD.TO": (0.08, 0.21, 0.045),
    "BNS.TO": (0.06, 0.22, 0.058),
    "BMO.TO": (0.08, 0.21, 0.046),
    "CM.TO": (0.07, 0.23, 0.052),
}

_SECTOR_LOADING = 0.55
_IDIO_LOADING = 0.30
_FACTOR_NORM = float(np.hypot(_SECTOR_LOADING, _IDIO_LOADING))

TRADING_DAYS = 252


class DataUnavailable(RuntimeError):
    pass


def _fetch_prices(yf, tickers: list[str], start: str, end: str | None,
                  attempts: int = 3, pause: float = 2.0) -> pd.DataFrame:
    last_exc: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            last_exc = exc
            log.warning("yf.download attempt %d/%d failed: %r", attempt, attempts, exc)
        else:
            prices = _extract_close(raw)
            if prices is not None and not prices.empty:
                return prices
            last_exc = DataUnavailable("download returned no usable rows")
            log.warning("yf.download attempt %d/%d returned nothing usable", attempt, attempts)

        if attempt < attempts:
            time.sleep(pause * attempt)

    raise DataUnavailable(f"price download failed after {attempts} attempts") from last_exc


def _extract_close(raw: pd.DataFrame) -> pd.DataFrame | None:
    if raw is None or raw.empty:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            return None
        prices = raw["Close"]
    elif "Close" in raw.columns:
        prices = raw[["Close"]]
    else:
        prices = raw

    return prices.dropna(how="all").ffill().dropna()


def _trailing_yield(yf, ticker: str, last_close: float) -> float:
    try:
        divs = yf.Ticker(ticker).dividends
        if divs is None or divs.empty:
            log.warning("no dividend history for %s", ticker)
            return float("nan")

        cutoff = divs.index.max() - pd.Timedelta(days=365)
        recent = divs[divs.index >= cutoff]
        if recent.empty or not np.isfinite(last_close) or last_close <= 0:
            return float("nan")

        return float(recent.sum() / last_close)
    except Exception as exc:
        log.warning("dividend lookup failed for %s: %r", ticker, exc)
        return float("nan")


def _load_yfinance(start: str, end: str | None):
    import yfinance as yf

    tickers = list(BANKS)
    prices = _fetch_prices(yf, tickers, start, end)

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise DataUnavailable(f"tickers missing from download: {missing}")

    prices = prices[tickers]
    yields = {t: _trailing_yield(yf, t, prices[t].iloc[-1]) for t in tickers}

    return prices, yields, "yfinance"


def _simulate(start: str, end: str | None, seed: int = 6):
    end = end or pd.Timestamp.today().normalize().isoformat()
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    if n < 2:
        raise ValueError(f"date range {start} to {end} has too few business days")

    tickers = list(BANKS)
    mu = np.array([_PROFILE[t][0] for t in tickers])
    sigma = np.array([_PROFILE[t][1] for t in tickers])
    yields = {t: _PROFILE[t][2] for t in tickers}

    step = 1 / TRADING_DAYS

    sector = np.random.default_rng(seed).standard_normal((n, 1))
    idio = np.random.default_rng(seed + 1).standard_normal((n, len(tickers)))
    shocks = (_SECTOR_LOADING * sector + _IDIO_LOADING * idio) / _FACTOR_NORM

    daily = mu * step + sigma * np.sqrt(step) * shocks
    prices = 50 * np.exp(np.cumsum(daily, axis=0))

    return pd.DataFrame(prices, index=dates, columns=tickers), yields, "simulated"


def load_data(start: str = "2015-01-01", end: str | None = None,
              allow_fallback: bool = True):
    try:
        return _load_yfinance(start, end)
    except ImportError as exc:
        log.error("yfinance is not installed: %r", exc)
        if not allow_fallback:
            raise DataUnavailable("yfinance is not installed") from exc
    except Exception as exc:
        log.error("live data unavailable: %r", exc)
        if not allow_fallback:
            raise

    log.warning("falling back to simulated prices")
    return _simulate(start, end)
