from nba_breakout import fetch_multiple_seasons, detect_breakouts, BreakoutConfig
import pandas as pd


def main():

    print("\nNBA Breakout Player Detection\n")

    refresh = input("Refresh stats from Basketball-Reference? (y/n): ").lower() == "y"

    prev_season = 2024
    curr_season = 2025
    seasons = [prev_season, curr_season]

    df_all = fetch_multiple_seasons(seasons, cache_dir="data_cache", refresh=refresh)

    cfg = BreakoutConfig()

    results = detect_breakouts(df_all, prev_season, curr_season, cfg)

    leaderboard = results[[
        "Player",
        f"Tm_{curr_season}",
        f"Teams_{curr_season}",
        f"Pos_{curr_season}",
        "Breakout Score",
        "MPG Δ",
        "PPG Δ",
        "APG Δ",
        "RPG Δ",
        "SPG Δ",
        "FT% Δ",
        "FG% Δ",
        "3P% Δ"
    ]].copy()

    leaderboard = leaderboard.rename(columns={
        f"Tm_{curr_season}": "Team",
        f"Teams_{curr_season}": "Teams",
        f"Pos_{curr_season}": "Pos"
    })

    pd.set_option("display.width", 160)

    print(f"\nTop Breakouts: {prev_season} → {curr_season}\n")
    print(leaderboard.drop(columns=["Teams"]).head(15).to_string(index=False))

    filename = f"breakouts_{prev_season}_to_{curr_season}.csv"
    leaderboard.to_csv(filename, index=False)

    print(f"\nBreakout report saved as: {filename}")


if __name__ == "__main__":
    main()
