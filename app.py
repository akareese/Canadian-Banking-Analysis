from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data import BANKS, load_data

logging.basicConfig(level=logging.INFO)

TRADING_DAYS = 252
START_DATE = "2015-01-01"

COLORS = {
    "RY.TO": "#1F4E9C",
    "TD.TO": "#205E3B",
    "BNS.TO": "#E03131",
    "BMO.TO": "#399BEB",
    "CM.TO": "#8B0000",
}

STATIC = {"staticPlot": True, "displayModeBar": False}

st.set_page_config(
    page_title="Big Five Canadian Banks",
    page_icon="🍁",
    layout="wide",
)


@st.cache_data(ttl=60 * 60 * 6, show_spinner="Fetching bank data…")
def get_data(start: str):
    return load_data(start)


def annualise_return(series: pd.Series) -> float:
    years = len(series) / TRADING_DAYS
    if years <= 0:
        return np.nan
    return (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1


def pct(value: float, places: int = 1) -> str:
    return "n/a" if not np.isfinite(value) else f"{value * 100:.{places}f}%"


selected = list(BANKS)

st.title("Big Five Canadian Banks Analysis")
st.write(
    "Compares RBC, TD, Scotiabank, BMO, and CIBC on long-run returns, volatility, and dividend income."
)

prices_all, yields, source = get_data(START_DATE)
prices = prices_all[selected]
rets = prices.pct_change().dropna()
labels = {t: BANKS[t] for t in selected}

if source != "yfinance":
    st.warning(
        "Live market data was unavailable, so every figure below comes from a "
        "simulated price series. Treat it as illustrative, not factual.",
        icon="⚠️",
    )

missing_yields = [labels[t] for t in selected if not np.isfinite(yields.get(t, np.nan))]
if missing_yields:
    st.info(
        "Dividend history could not be retrieved for: "
        + ", ".join(missing_yields)
        + ". These are omitted from the yield chart rather than shown as zero."
    )

st.caption(
    f"Source: {source}. Period: {prices.index[0].date()} to {prices.index[-1].date()} "
    f"({len(prices):,} trading days)."
)

summary = pd.DataFrame({
    "Total Return": prices.iloc[-1] / prices.iloc[0] - 1,
    "Ann. Return": {t: annualise_return(prices[t]) for t in selected},
    "Ann. Volatility": rets.std() * np.sqrt(TRADING_DAYS),
    "Dividend Yield": pd.Series({t: yields.get(t, np.nan) for t in selected}),
})

best = summary["Total Return"].idxmax()
c1, c2, c3 = st.columns(3)
c1.metric("Best total return", labels[best],
          pct(summary.loc[best, "Total Return"], 0))

if summary["Dividend Yield"].notna().any():
    top_yield = summary["Dividend Yield"].idxmax()
    c2.metric("Highest dividend yield", labels[top_yield],
              pct(summary.loc[top_yield, "Dividend Yield"]))
else:
    c2.metric("Highest dividend yield", "n/a", "no data")

c3.metric("Avg. pairwise correlation",
          f"{rets.corr().where(~np.eye(len(selected), dtype=bool)).stack().mean():.2f}")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Investment Growth")
    growth = prices / prices.iloc[0]
    fig = go.Figure()

    for t in selected:
        fig.add_trace(
            go.Scatter(
                x=growth.index,
                y=growth[t],
                name=labels[t],
                line=dict(color=COLORS[t], width=2),
            )
        )

    fig.update_layout(
        yaxis_title="Value (start = $1)",
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title_text="",
        height=380,
    )

    st.plotly_chart(fig, config=STATIC, width="stretch")

    basis = (
        "Closes are dividend adjusted, so this is total shareholder return."
        if source == "yfinance"
        else "Simulated prices exclude dividends, so this is price return only."
    )
    st.caption(
        "Tracks the cumulative growth of a $1 investment since 2015, allowing "
        "comparison of long-term shareholder returns across Canada's Big Five banks. "
        + basis
    )

with right:
    st.subheader("Risk vs Return (Annualized)")
    fig = go.Figure()

    for t in selected:
        fig.add_trace(
            go.Scatter(
                x=[summary.loc[t, "Ann. Volatility"] * 100],
                y=[summary.loc[t, "Ann. Return"] * 100],
                mode="markers+text",
                name=labels[t],
                text=[labels[t]],
                textposition="top center",
                marker=dict(size=16, color=COLORS[t]),
                showlegend=False,
            )
        )

    fig.update_layout(
        xaxis_title="Volatility (%)",
        yaxis_title="Return (%)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=380,
    )

    st.plotly_chart(fig, config=STATIC, width="stretch")

    st.caption(
        "Evaluates each bank's risk-return profile using annualized volatility and returns. "
        "Higher returns with lower volatility indicate stronger performance."
    )

left2, right2 = st.columns(2)

with left2:
    st.subheader("Trailing Dividend Yield")

    have_yield = [t for t in selected if np.isfinite(yields.get(t, np.nan))]

    if have_yield:
        yvals = [yields[t] * 100 for t in have_yield]

        fig = go.Figure(
            go.Bar(
                x=[labels[t] for t in have_yield],
                y=yvals,
                marker_color=[COLORS[t] for t in have_yield],
                text=[f"{v:.1f}%" for v in yvals],
                textposition="outside",
            )
        )

        fig.update_layout(
            yaxis_title="Yield (%)",
            margin=dict(l=10, r=10, t=10, b=10),
            height=360,
        )

        st.plotly_chart(fig, config=STATIC, width="stretch")
    else:
        st.error("No dividend data available for any ticker.")

    st.caption(
        "Compares trailing dividend yields to assess the income potential offered "
        "by each bank's stock."
    )

with right2:
    st.subheader("Daily Return Correlation")

    corr = rets.corr()
    short = [labels[t].split("(")[0].strip() for t in corr.columns]

    fig = px.imshow(
        corr.values,
        x=short,
        y=short,
        color_continuous_scale="YlGnBu",
        zmin=corr.values.min(),
        zmax=1,
        text_auto=".2f",
        aspect="auto",
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=360,
    )

    st.plotly_chart(fig, config=STATIC, width="stretch")

    st.caption(
        "Shows the relationship between daily stock returns. High correlations "
        "suggest the banks tend to react similarly to market conditions."
    )

st.divider()
