from __future__ import annotations

import argparse
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import BANKS, DataUnavailable, load_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

TRADING_DAYS = 252
START_DATE = "2015-01-01"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})


COLORS = {
    "RY.TO": "#1F4E9C",
    "TD.TO": "#205E3B",
    "BNS.TO": "#E03131",
    "BMO.TO": "#399BEB",
    "CM.TO": "#8B0000",
}


def annualise_return(prices: pd.Series) -> float:
    years = len(prices) / TRADING_DAYS
    if years <= 0:
        return np.nan
    return (prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1


def fmt_pct(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value * 100:.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description="Big Five Canadian banks comparison")
    parser.add_argument("--start", default=START_DATE, help="first date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None,
                        help="last date (YYYY-MM-DD); defaults to the latest session")
    parser.add_argument("--no-fallback", action="store_true",
                        help="fail instead of falling back to simulated prices")
    parser.add_argument("--out", default="banks_showdown.png", help="output image path")
    args = parser.parse_args()

    try:
        prices, div_yields, source = load_data(
            args.start, args.end, allow_fallback=not args.no_fallback
        )
    except DataUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    rets = prices.pct_change().dropna()
    labels = {t: BANKS[t] for t in prices.columns}

    summary = pd.DataFrame({
        "Total Return": prices.iloc[-1] / prices.iloc[0] - 1,
        "Ann. Return": {t: annualise_return(prices[t]) for t in prices.columns},
        "Ann. Volatility": rets.std() * np.sqrt(TRADING_DAYS),
        "Dividend Yield": pd.Series(div_yields),
    })
    summary.index = [labels[t] for t in summary.index]

    print(f"\nData source: {source}")
    if source != "yfinance":
        print("WARNING: figures below are simulated, not real market data.")
    print(f"Period: {prices.index[0].date()} -> {prices.index[-1].date()} "
          f"({len(prices):,} trading days)\n")
    print("Big Five Canadian Banks - comparison")
    print("=" * 64)
    print(summary.map(fmt_pct).to_string())
    print("=" * 64)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Big Five Canadian Banks  ({source} data, "
                 f"{prices.index[0].year}-{prices.index[-1].year})",
                 fontsize=15, fontweight="bold", y=0.98)

    ax = axes[0, 0]
    growth = prices / prices.iloc[0]
    for t in prices.columns:
        ax.plot(growth[t], label=labels[t], color=COLORS[t], lw=1.8)
    ax.set_title("Growth of One Dollar Invested")
    ax.set_ylabel("Value (start = 1.0)")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[0, 1]
    for t in prices.columns:
        x = summary.loc[labels[t], "Ann. Volatility"] * 100
        y = summary.loc[labels[t], "Ann. Return"] * 100
        ax.scatter(x, y, s=140, color=COLORS[t], zorder=3)
        ax.annotate(labels[t].split("(")[0].strip(), (x, y),
                    xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set_title("Risk vs Return (Annualised)")
    ax.set_xlabel("Volatility (%)")
    ax.set_ylabel("Return (%)")

    ax = axes[1, 0]
    have = [t for t in prices.columns if np.isfinite(div_yields.get(t, np.nan))]
    if have:
        yvals = [div_yields[t] * 100 for t in have]
        bars = ax.bar([labels[t] for t in have], yvals,
                      color=[COLORS[t] for t in have])
        for bar, v in zip(bars, yvals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                    f"{v:.1f}%", ha="center", fontsize=8)
    else:
        ax.text(0.5, 0.5, "no dividend data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10)
    ax.set_title("Trailing Dividend Yield")
    ax.set_ylabel("Yield (%)")
    ax.tick_params(axis="x", rotation=25)

    ax = axes[1, 1]
    corr = rets.corr()
    corr.index = [labels[t].split("(")[0].strip() for t in corr.index]
    corr.columns = corr.index
    im = ax.imshow(corr, cmap="YlGnBu", vmin=corr.values.min(), vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black")
    ax.set_title("Daily-Return Correlation")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nSaved chart pack -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
