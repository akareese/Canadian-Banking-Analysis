import re
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="NBA Breakout Analysis", page_icon="🏀", layout="wide")

MULTI_TEAM_RE = re.compile(r"^(TOT|\d+TM)$")

st.html("""
<style>
.st-key-score_slider [data-testid="stSliderThumbValue"],
.st-key-score_slider [data-testid="stThumbValue"],
.st-key-score_slider div[class*="StyledThumbValue"] {
    opacity: 0;
    transition: opacity 120ms ease;
}
.st-key-score_slider:hover [data-testid="stSliderThumbValue"],
.st-key-score_slider:hover [data-testid="stThumbValue"],
.st-key-score_slider:hover div[class*="StyledThumbValue"],
.st-key-score_slider:focus-within [data-testid="stSliderThumbValue"],
.st-key-score_slider:focus-within [data-testid="stThumbValue"],
.st-key-score_slider:focus-within div[class*="StyledThumbValue"] {
    opacity: 1;
}
.st-key-score_slider [data-testid="stSliderTickBarMin"],
.st-key-score_slider [data-testid="stSliderTickBarMax"],
.st-key-score_slider [data-testid="stTickBarMin"],
.st-key-score_slider [data-testid="stTickBarMax"] {
    font-size: 0 !important;
}
.st-key-score_slider [data-testid="stSliderTickBarMin"]::after,
.st-key-score_slider [data-testid="stTickBarMin"]::after {
    content: "Min";
    font-size: 0.75rem;
}
.st-key-score_slider [data-testid="stSliderTickBarMax"]::after,
.st-key-score_slider [data-testid="stTickBarMax"]::after {
    content: "Max";
    font-size: 0.75rem;
}
.leaderboard-wrap {
    max-height: 620px;
    overflow: auto;
    border: 1px solid rgba(128, 128, 128, 0.3);
    border-radius: 8px;
}
.leaderboard-wrap table {
    border-collapse: collapse;
    width: 100%;
    font-size: 0.85rem;
    font-variant-numeric: tabular-nums;
}
.leaderboard-wrap thead th {
    position: sticky;
    top: 0;
    background: var(--secondary-background-color, #f0f2f6);
    color: inherit;
    text-align: right;
    padding: 8px 10px;
    border-bottom: 1px solid rgba(128, 128, 128, 0.35);
    white-space: nowrap;
    z-index: 2;
}
.leaderboard-wrap tbody td {
    padding: 6px 10px;
    text-align: right;
    border-bottom: 1px solid rgba(128, 128, 128, 0.15);
    white-space: nowrap;
}
.leaderboard-wrap tbody td:nth-child(1),
.leaderboard-wrap thead th:nth-child(1) {
    text-align: left;
}
.leaderboard-wrap tbody tr:hover td {
    background: rgba(128, 128, 128, 0.08);
}
.leaderboard-wrap td[title] {
    cursor: help;
    text-decoration: underline dotted currentColor;
    text-underline-offset: 3px;
}
</style>
""")

st.markdown("""
    <h1 style='text-align: center;'>NBA Breakout Player Detection</h1>
    <p style='text-align: center; color: gray; font-size: 1.1rem;'>2024 → 2025 Season Comparison</p>
    <hr>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_csv("breakouts_2024_to_2025.csv")


df = load_data()

st.sidebar.header("Filters")

positions = ["All"] + sorted(df["Pos"].dropna().unique().tolist())
selected_pos = st.sidebar.selectbox("Position", positions)

teams = ["All"] + sorted(df["Team"].dropna().unique().tolist())
selected_team = st.sidebar.selectbox("Team", teams)

min_score, max_score = float(df["Breakout Score"].min()), float(df["Breakout Score"].max())

with st.sidebar.container(key="score_slider"):
    score_range = st.slider(
        "Breakout Score Range",
        min_value=min_score,
        max_value=max_score,
        value=(min_score, max_score),
        step=0.1,
        format="%.1f",
    )

top_n = st.sidebar.slider("Show Top N Players", min_value=5, max_value=len(df), value=50, step=5)

filtered = df.copy()
if selected_pos != "All":
    filtered = filtered[filtered["Pos"] == selected_pos]
if selected_team != "All":
    filtered = filtered[filtered["Team"] == selected_team]
filtered = filtered[
    (filtered["Breakout Score"] >= score_range[0]) &
    (filtered["Breakout Score"] <= score_range[1])
]
filtered = filtered.head(top_n).reset_index(drop=True)

col1, col2, col3 = st.columns(3)
col1.metric("Players Shown", len(filtered))
col2.metric("Top Breakout Score", f"{filtered['Breakout Score'].max():.1f}" if not filtered.empty else "—")
col3.metric("Avg Breakout Score", f"{filtered['Breakout Score'].mean():.2f}" if not filtered.empty else "—")

st.markdown("### 📊 Breakout Leaderboard")
st.caption("Team cells marked 2TM or 3TM are underlined. Hover one to see the teams in the order the player joined them.")

delta_cols = ["MPG Δ", "PPG Δ", "APG Δ", "RPG Δ", "SPG Δ", "FT% Δ", "FG% Δ", "3P% Δ"]
present_delta_cols = [c for c in delta_cols if c in filtered.columns]


def style_table(row):
    styles = [""] * len(row)
    idx = row.index.tolist()
    if "Breakout Score" in idx:
        val = row["Breakout Score"]
        i = idx.index("Breakout Score")
        if val > 3:
            styles[i] = "background-color: #1a7a3a; color: white; font-weight: bold;"
        elif val > 1:
            styles[i] = "background-color: #a07800; color: white; font-weight: bold;"
        elif val < 0:
            styles[i] = "background-color: #8b1a1a; color: white; font-weight: bold;"
        else:
            styles[i] = "background-color: #2a5a2a; color: white; font-weight: bold;"
    for col in present_delta_cols:
        if col in idx:
            val = row[col]
            i = idx.index(col)
            try:
                if val > 0:
                    styles[i] = "color: #4caf50;"
                elif val < 0:
                    styles[i] = "color: #f44336;"
            except TypeError:
                pass
    return styles


def split_teams(value):
    if pd.isna(value):
        return []
    return [t.strip() for t in str(value).replace("→", ",").split(",") if t.strip()]


def build_tooltips(display_df, source_df):
    ttips = pd.DataFrame("", index=display_df.index, columns=display_df.columns)
    if "Team" not in display_df.columns or "Teams" not in source_df.columns:
        return ttips
    for i in display_df.index:
        code = str(display_df.at[i, "Team"]).strip().upper()
        if not MULTI_TEAM_RE.match(code):
            continue
        seq = split_teams(source_df.at[i, "Teams"])
        if len(seq) > 1:
            ttips.at[i, "Team"] = "Joined in order: " + " → ".join(seq)
    return ttips


display = filtered.drop(columns=["Teams"], errors="ignore")

fmt = {col: "{:+.1f}" for col in present_delta_cols}
fmt["Breakout Score"] = "{:.1f}"

if display.empty:
    st.info("No players match the current filters.")
else:
    ttips = build_tooltips(display, filtered)

    styled = (
        display.style
        .apply(style_table, axis=1)
        .format(fmt, na_rep="—")
        .set_tooltips(ttips, as_title_attribute=True)
        .hide(axis="index")
    )

    st.html(f'<div class="leaderboard-wrap">{styled.to_html()}</div>')

    with st.expander("Sortable view"):
        st.dataframe(display, use_container_width=True, hide_index=True)

st.markdown("### 📈 Breakout Score Chart")
st.markdown("""
> Each bar represents a player's **Breakout Score**, a composite metric that measures 
> how much a player improved from the 2024 season to the 2025 season. The score is calculated using a weighted 
> formula that factors in Points per 36 minutes (55%), True Shooting % (30%), Assists per 36 minutes (10%), 
> and Turnovers per 36 minutes (5%). A **higher score means a stronger breakout season**. Players with a score 
> above 3 are considered standout breakouts, while negative scores indicate a statistical decline.
""")

chart_data = filtered.sort_values("Breakout Score", ascending=True)

hover_team = [
    " → ".join(split_teams(t)) if MULTI_TEAM_RE.match(str(c).strip().upper()) else str(c)
    for c, t in zip(chart_data.get("Team", []), chart_data.get("Teams", chart_data.get("Team", [])))
]

fig = go.Figure(go.Bar(
    x=chart_data["Player"],
    y=chart_data["Breakout Score"],
    marker_color="#4c9be8",
    customdata=hover_team,
    hovertemplate="<b>%{x}</b><br>%{customdata}<br>Score: %{y:.1f}<extra></extra>",
))

fig.update_layout(
    xaxis_tickangle=-45,
    xaxis_title=None,
    yaxis_title="Breakout Score",
    dragmode=False,
    height=500,
    margin=dict(t=20, b=120),
)

fig.update_layout(
    modebar_remove=["zoom", "pan", "select", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"]
)

st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False, "displayModeBar": False})
