import re
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass

BASE_URL = "https://www.basketball-reference.com/leagues/NBA_{season}_per_game.html"

MULTI_TEAM_RE = re.compile(r"^\d+TM$")


@dataclass
class BreakoutConfig:
    min_games_prev: int = 30
    min_games_curr: int = 30
    min_minutes_prev: int = 800
    min_minutes_curr: int = 800

    w_pts_per36: float = 0.55
    w_ts: float = 0.30
    w_ast_per36: float = 0.10
    w_tov_per36: float = 0.05


def fetch_season(season, cache_dir="data_cache", refresh=False):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)

    csv_path = cache_dir / f"NBA_{season}_per_game.csv"
    if csv_path.exists() and not refresh:
        return pd.read_csv(csv_path)

    url = BASE_URL.format(season=season)
    tables = pd.read_html(url)
    df = tables[0]

    if "Rk" in df.columns:
        df = df[df["Rk"].astype(str) != "Rk"]

    df.columns = df.columns.str.strip()

    if "Team" in df.columns and "Tm" not in df.columns:
        df = df.rename(columns={"Team": "Tm"})

    df = df.dropna(subset=["Player"])

    exclude_cols = ["Player", "Pos", "Tm"]
    numeric_cols = [c for c in df.columns if c not in exclude_cols]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    df["Season"] = season
    df["SourceRow"] = np.arange(len(df))

    df.to_csv(csv_path, index=False)
    return df


def fetch_multiple_seasons(seasons, cache_dir="data_cache", refresh=False):
    dfs = [fetch_season(s, cache_dir, refresh) for s in seasons]
    return pd.concat(dfs, ignore_index=True)


def is_multi_team(series):
    v = series.astype(str).str.strip().str.upper()
    return v.eq("TOT") | v.str.match(MULTI_TEAM_RE)


def add_team_sequence(df):
    df = df.copy()

    if "Tm" not in df.columns:
        df["Teams"] = np.nan
        return df

    if "SourceRow" not in df.columns:
        df["SourceRow"] = np.arange(len(df))

    singles = df.loc[~is_multi_team(df["Tm"])].sort_values(
        ["Player", "Season", "SourceRow"]
    )

    seq = (
        singles.groupby(["Player", "Season"])["Tm"]
        .apply(lambda s: ",".join(dict.fromkeys(s.astype(str).str.strip())))
    )

    key = pd.MultiIndex.from_arrays([df["Player"], df["Season"]])
    df["Teams"] = seq.reindex(key).to_numpy()
    df["Teams"] = df["Teams"].fillna(df["Tm"].astype(str).str.strip())

    return df


def pick_tot_rows(df):
    if "Tm" not in df.columns:
        return df

    df = add_team_sequence(df)
    df["is_tot"] = is_multi_team(df["Tm"]).astype(int)
    df = df.sort_values(
        ["Player", "Season", "is_tot", "G"], ascending=[True, True, False, False]
    )
    df = df.drop_duplicates(["Player", "Season"], keep="first")
    return df.drop(columns=["is_tot"])


def add_features(df):
    df = df.copy()

    df["MIN_total"] = (df["MP"] * df["G"]).round(1)

    denom = 2 * (df["FGA"] + 0.44 * df["FTA"])
    df["TS%"] = np.where(denom > 0, df["PTS"] / denom, np.nan)

    df["PTS_per36"] = np.where(df["MP"] > 0, df["PTS"] * 36 / df["MP"], np.nan)
    df["AST_per36"] = np.where(df["MP"] > 0, df["AST"] * 36 / df["MP"], np.nan)
    df["TOV_per36"] = np.where(df["MP"] > 0, df["TOV"] * 36 / df["MP"], np.nan)

    return df


def _pct_point_change(curr: pd.Series, prev: pd.Series) -> pd.Series:
    return ((curr - prev) * 100.0).round(1)


def detect_breakouts(df_all, prev_season, curr_season, cfg: BreakoutConfig):
    df = df_all[df_all["Season"].isin([prev_season, curr_season])].copy()
    df = pick_tot_rows(df)
    df = add_features(df)

    keep_cols = [
        "Player", "Season", "Pos", "Tm", "Teams", "G", "MIN_total",
        "MP", "PTS", "AST", "TRB", "STL",
        "FT%", "FG%", "3P%",
        "PTS_per36", "AST_per36", "TOV_per36", "TS%"
    ]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    prev = df[df["Season"] == prev_season].copy()
    curr = df[df["Season"] == curr_season].copy()

    prev = prev[(prev["G"] >= cfg.min_games_prev) & (prev["MIN_total"] >= cfg.min_minutes_prev)]
    curr = curr[(curr["G"] >= cfg.min_games_curr) & (curr["MIN_total"] >= cfg.min_minutes_curr)]

    merged = prev.merge(curr, on="Player", suffixes=(f"_{prev_season}", f"_{curr_season}"))

    merged["MPG Δ"] = (merged[f"MP_{curr_season}"] - merged[f"MP_{prev_season}"]).round(1)
    merged["PPG Δ"] = (merged[f"PTS_{curr_season}"] - merged[f"PTS_{prev_season}"]).round(1)
    merged["APG Δ"] = (merged[f"AST_{curr_season}"] - merged[f"AST_{prev_season}"]).round(1)

    if f"TRB_{prev_season}" in merged.columns and f"TRB_{curr_season}" in merged.columns:
        merged["RPG Δ"] = (merged[f"TRB_{curr_season}"] - merged[f"TRB_{prev_season}"]).round(1)
    else:
        merged["RPG Δ"] = np.nan

    if f"STL_{prev_season}" in merged.columns and f"STL_{curr_season}" in merged.columns:
        merged["SPG Δ"] = (merged[f"STL_{curr_season}"] - merged[f"STL_{prev_season}"]).round(1)
    else:
        merged["SPG Δ"] = np.nan

    merged["FT% Δ"] = _pct_point_change(merged.get(f"FT%_{curr_season}"), merged.get(f"FT%_{prev_season}"))
    merged["FG% Δ"] = _pct_point_change(merged.get(f"FG%_{curr_season}"), merged.get(f"FG%_{prev_season}"))
    merged["3P% Δ"] = _pct_point_change(merged.get(f"3P%_{curr_season}"), merged.get(f"3P%_{prev_season}"))

    d_pts36 = merged[f"PTS_per36_{curr_season}"] - merged[f"PTS_per36_{prev_season}"]
    d_ast36 = merged[f"AST_per36_{curr_season}"] - merged[f"AST_per36_{prev_season}"]
    d_ts = merged[f"TS%_{curr_season}"] - merged[f"TS%_{prev_season}"]
    d_tov36 = merged[f"TOV_per36_{curr_season}"] - merged[f"TOV_per36_{prev_season}"]

    merged["Breakout Score"] = (
        cfg.w_pts_per36 * d_pts36
        + cfg.w_ts * (d_ts * 100.0)
        + cfg.w_ast_per36 * d_ast36
        - cfg.w_tov_per36 * d_tov36
    ).round(1)

    merged = merged.sort_values("Breakout Score", ascending=False).reset_index(drop=True)
    return merged
