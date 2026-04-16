"""Production rating + prediction.

Fits the recommended model (ridge regression with recency weighting) on the
full filtered dataset and writes ``data/ratings.csv``. Provides a tidy DataFrame
for downstream code and a ``predict_game`` helper for ad-hoc matchups.

CLI:
    uv run python -m softballratings.rate              # use cached games
    uv run python -m softballratings.rate --refresh    # re-scrape Massey
    uv run python -m softballratings.rate --top 50     # show more teams
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

    # P(team beats league-average opponent on a neutral field)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build production softball ratings.")
    parser.add_argument("--refresh", action="store_true", help="re-scrape Massey")
    parser.add_argument("--out", default=str(DEFAULT_RATINGS_PATH))
    parser.add_argument("--lam", type=float, default=PROD_LAM)
    parser.add_argument("--half-life", type=float, default=PROD_HALF_LIFE)
    parser.add_argument("--top", type=int, default=25, help="how many teams to print")
    args = parser.parse_args()

    games = filter_to_core(fetch_raw(refresh=args.refresh))
    df = build_ratings(
        games,
        lam=args.lam,
        recency_half_life=args.half_life,
        save_to=Path(args.out),
    )

    print(f"fit on {len(games)} games, {len(df)} teams")
    print(f"HFA = {df['hfa'].iloc[0]:.3f} runs    sigma = {df['sigma'].iloc[0]:.3f}    league_mean = {df['league_mean'].iloc[0]:.3f}")
    print(f"\nTop {args.top}:")
    cols = ["rank", "team", "n_games", "off", "def", "net", "p_beat_avg"]
    print(df[cols].head(args.top).round(3).to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
