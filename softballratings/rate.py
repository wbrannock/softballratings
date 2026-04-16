"""Production rating + prediction.

Fits the recommended model (ridge regression with recency weighting) on the
full filtered dataset and writes ``data/ratings.csv``. Provides a tidy DataFrame
for downstream code and a ``predict_game`` helper for ad-hoc matchups.

Daily workflow:

    uv run python -m softballratings.rate          # auto-refresh if cache > 6h old
    uv run python -m softballratings.rate --top 50
    uv run python -m softballratings.rate --refresh                # force re-scrape
    uv run python -m softballratings.rate --max-age 0              # same as --refresh
    uv run python -m softballratings.rate --no-save                # don't touch CSVs

Each run prints the top-N table and "biggest movers since last run" by comparing
against the previous saved ``data/ratings.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .ratings import RatingsResult, fit_ratings_ridge
from .scrape import fetch_raw, filter_to_core

DEFAULT_RATINGS_PATH = Path("data/ratings.csv")
PREV_RATINGS_PATH = Path("data/ratings_prev.csv")
DEFAULT_MAX_AGE_HOURS = 6.0

# Chosen via the eval harness: best log-loss/Brier among all variants tried,
# within a hair of the best MAE, simple to fit, and adapts as the season ages.
PROD_LAM = 3.0
PROD_HALF_LIFE = 45.0


def _residual_sigma(ratings: RatingsResult, games: pd.DataFrame) -> float:
    """Std of the model's margin residuals on its own training data — used
    as the noise scale for win-probability calculations."""
    off, def_ = ratings.off, ratings.def_
    home = games["home_team"].to_numpy()
    away = games["away_team"].to_numpy()
    neutral = games["neutral"].to_numpy(dtype=bool)
    pred = (
        (off.loc[home].to_numpy() - off.loc[away].to_numpy())
        + (def_.loc[away].to_numpy() - def_.loc[home].to_numpy())
        + np.where(neutral, 0.0, ratings.hfa)
    )
    actual = (games["home_score"] - games["away_score"]).to_numpy(dtype=float)
    return float(np.std(pred - actual, ddof=1))


def build_ratings(
    games: pd.DataFrame | None = None,
    lam: float = PROD_LAM,
    recency_half_life: float | None = PROD_HALF_LIFE,
    blowout_cap: float | None = None,
    save_to: Path | None = DEFAULT_RATINGS_PATH,
) -> pd.DataFrame:
    """Fit the production rating model and return a tidy DataFrame.

    Columns: rank, team, n_games, off, def, net, p_beat_avg, hfa, league_mean, sigma.
    """
    if games is None:
        games = filter_to_core(fetch_raw())

    r = fit_ratings_ridge(
        games,
        lam=lam,
        recency_half_life=recency_half_life,
        blowout_cap=blowout_cap,
    )

    counts = pd.concat([games["home_team"], games["away_team"]]).value_counts()
    sigma = _residual_sigma(r, games)
    p_avg = norm.cdf((r.off - r.def_).to_numpy() / sigma)

    df = pd.DataFrame({
        "team": r.off.index,
        "n_games": counts.reindex(r.off.index).to_numpy(),
        "off": r.off.to_numpy(),
        "def": r.def_.to_numpy(),
        "net": (r.off - r.def_).to_numpy(),
        "p_beat_avg": p_avg,
    }).sort_values("net", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    df["hfa"] = r.hfa
    df["league_mean"] = r.league_mean
    df["sigma"] = sigma

    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_to, index=False)
    return df


def predict_game(ratings: pd.DataFrame, home: str, away: str, neutral: bool = False) -> dict:
    """Predict a head-to-head from a ratings DataFrame produced by build_ratings."""
    by_team = ratings.set_index("team")
    for t in (home, away):
        if t not in by_team.index:
            raise KeyError(f"unknown team: {t!r}")

    h = by_team.loc[home]
    a = by_team.loc[away]
    hfa = float(h["hfa"])
    mu = float(h["league_mean"])
    sigma = float(h["sigma"])
    half_hfa = 0.0 if neutral else 0.5 * hfa

    pred_margin = (h["off"] - a["off"]) + (a["def"] - h["def"]) + (0.0 if neutral else hfa)
    pred_home = mu + h["off"] + a["def"] + half_hfa
    pred_away = mu + a["off"] + h["def"] - half_hfa

    return {
        "home": home,
        "away": away,
        "neutral": neutral,
        "pred_home_score": float(pred_home),
        "pred_away_score": float(pred_away),
        "pred_margin": float(pred_margin),
        "p_home_wins": float(norm.cdf(pred_margin / sigma)),
    }


def compute_movers(new: pd.DataFrame, old: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Teams whose rank changed the most between two ratings runs."""
    if old is None or old.empty:
        return pd.DataFrame()
    merged = new[["team", "rank", "net"]].merge(
        old[["team", "rank", "net"]],
        on="team",
        suffixes=("_new", "_old"),
    )
    merged["rank_change"] = merged["rank_old"] - merged["rank_new"]   # positive = moved up
    merged["net_change"] = merged["net_new"] - merged["net_old"]
    movers = merged.reindex(merged["rank_change"].abs().sort_values(ascending=False).index)
    return movers.head(top_n).reset_index(drop=True)


def _format_movers(movers: pd.DataFrame) -> str:
    if movers.empty:
        return "  (no previous run to compare against)"
    lines = []
    for _, row in movers.iterrows():
        if row["rank_change"] == 0:
            continue
        arrow = "↑" if row["rank_change"] > 0 else "↓"
        lines.append(
            f"  {arrow} {row['team']:<18} "
            f"rank {int(row['rank_old']):>3} → {int(row['rank_new']):<3}  "
            f"({row['rank_change']:+d}, net {row['net_change']:+.2f})"
        )
    return "\n".join(lines) if lines else "  (no rank changes)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build current softball ratings (auto-refreshes stale data)."
    )
    parser.add_argument("--refresh", action="store_true", help="force re-scrape regardless of cache age")
    parser.add_argument("--max-age", type=float, default=DEFAULT_MAX_AGE_HOURS,
                        help=f"auto-refresh if cache is older than this many hours (default {DEFAULT_MAX_AGE_HOURS})")
    parser.add_argument("--out", default=str(DEFAULT_RATINGS_PATH))
    parser.add_argument("--lam", type=float, default=PROD_LAM)
    parser.add_argument("--half-life", type=float, default=PROD_HALF_LIFE)
    parser.add_argument("--blowout-cap", type=float, default=None)
    parser.add_argument("--top", type=int, default=25, help="how many teams to print")
    parser.add_argument("--no-save", action="store_true", help="don't write any files")
    parser.add_argument("--no-movers", action="store_true", help="skip the movers comparison")
    args = parser.parse_args()

    out_path = Path(args.out)

    # Load previous ratings BEFORE we overwrite the file, so we can compute movers.
    prev_df: pd.DataFrame | None = None
    if not args.no_movers and out_path.exists():
        try:
            prev_df = pd.read_csv(out_path)
        except Exception:
            prev_df = None

    games = filter_to_core(
        fetch_raw(refresh=args.refresh, max_age_hours=args.max_age)
    )

    df = build_ratings(
        games,
        lam=args.lam,
        recency_half_life=args.half_life,
        blowout_cap=args.blowout_cap,
        save_to=None,  # we'll handle saving below to keep the prev/current rotation clean
    )

    # Persist
    if not args.no_save:
        if prev_df is not None:
            PREV_RATINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            prev_df.to_csv(PREV_RATINGS_PATH, index=False)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

    # ---- display ----
    print(f"fit on {len(games)} games, {len(df)} teams "
          f"({games['date'].min().date()} → {games['date'].max().date()})")
    print(f"HFA = {df['hfa'].iloc[0]:.3f} runs    "
          f"sigma = {df['sigma'].iloc[0]:.3f}    "
          f"league_mean = {df['league_mean'].iloc[0]:.3f}")

    print(f"\nTop {args.top}:")
    cols = ["rank", "team", "n_games", "off", "def", "net", "p_beat_avg"]
    print(df[cols].head(args.top).round(3).to_string(index=False))

    if not args.no_movers:
        print("\nBiggest movers since last run:")
        print(_format_movers(compute_movers(df, prev_df, top_n=10)))

    if not args.no_save:
        print(f"\nwrote {out_path}"
              + (f" (previous → {PREV_RATINGS_PATH})" if prev_df is not None else ""))


if __name__ == "__main__":
    main()
