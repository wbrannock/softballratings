"""Eval harness for rating models.

Library + CLI. Usage:

    uv run python -m softballratings.eval                  # run all models
    uv run python -m softballratings.eval --models ridge:lam=3
    uv run python -m softballratings.eval --n-test 600
    uv run python -m softballratings.eval --no-save
    uv run python -m softballratings.eval --history       # best-ever per model
    uv run python -m softballratings.eval --list

Results land in ``data/eval_history.csv``. Add a model to ``MODELS`` and rerun
to compare it against everything that came before.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm

from .ratings import RatingsResult, fit_ratings, fit_ratings_adaptive_ridge, fit_ratings_ridge
from .scrape import fetch_raw, filter_to_core

HISTORY_PATH = Path("data/eval_history.csv")


# ===================================================================
# Core data structures
# ===================================================================

@dataclass
class Predictions:
    """Output of a single model on a test set, plus its train-side noise."""
    pred_margin: np.ndarray   # predicted home_score - away_score
    pred_home: np.ndarray     # predicted home_score (may be NaN if model can't predict scores)
    pred_away: np.ndarray     # predicted away_score
    mask: np.ndarray          # True where both teams were known at train time
    sigma: float              # std of (pred - actual) margin residuals on the training set


@dataclass
class EvalMetrics:
    n: int
    margin_mae: float
    margin_rmse: float
    margin_bias: float
    win_acc: float
    log_loss: float           # binary cross-entropy on home-win probability
    brier: float              # Brier score on home-win probability
    home_mae: float           # per-side run prediction error (NaN if not supported)
    away_mae: float
    n_unrated: int
    sigma: float


# ===================================================================
# Split + central metric computation
# ===================================================================

def chrono_split(games: pd.DataFrame, n_test: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = games.sort_values("date", kind="stable").reset_index(drop=True)
    return g.iloc[:-n_test].reset_index(drop=True), g.iloc[-n_test:].reset_index(drop=True)


def _safe_log(p: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return np.log(np.clip(p, eps, 1 - eps))


def metrics_from_predictions(preds: Predictions, test: pd.DataFrame) -> EvalMetrics:
    actual_margin = (test["home_score"] - test["away_score"]).to_numpy(dtype=float)
    actual_home = test["home_score"].to_numpy(dtype=float)
    actual_away = test["away_score"].to_numpy(dtype=float)
    home_won = (actual_margin > 0).astype(float)

    m = preds.mask
    n = int(m.sum())

    err = preds.pred_margin[m] - actual_margin[m]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))

    pred_w = np.sign(preds.pred_margin[m])
    actual_w = np.sign(actual_margin[m])
    win_acc = float(np.mean((pred_w == actual_w) | ((pred_w == 0) & (actual_w == 0))))

    # Implied home-win probability via Normal(pred_margin, sigma).
    sigma = max(preds.sigma, 1e-6)
    p_home = norm.cdf(preds.pred_margin[m] / sigma)
    y = home_won[m]
    log_loss = float(-np.mean(y * _safe_log(p_home) + (1 - y) * _safe_log(1 - p_home)))
    brier = float(np.mean((p_home - y) ** 2))

    if not np.isnan(preds.pred_home).all():
        home_mae = float(np.nanmean(np.abs(preds.pred_home[m] - actual_home[m])))
        away_mae = float(np.nanmean(np.abs(preds.pred_away[m] - actual_away[m])))
    else:
        home_mae = float("nan")
        away_mae = float("nan")

    return EvalMetrics(
        n=n,
        margin_mae=mae,
        margin_rmse=rmse,
        margin_bias=bias,
        win_acc=win_acc,
        log_loss=log_loss,
        brier=brier,
        home_mae=home_mae,
        away_mae=away_mae,
        n_unrated=int((~m).sum()),
        sigma=sigma,
    )


# ===================================================================
# Ratings-based predictor (used by both iter_mean and ridge)
# ===================================================================

def _ratings_score_preds(ratings: RatingsResult, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pred_home_score, pred_away_score, mask) using the rating model
    home_score = mu + Off[h] + Def[a] + 0.5*HFA*nn, etc.
    """
    off, def_ = ratings.off, ratings.def_
    known = set(off.index)
    home = df["home_team"].to_numpy()
    away = df["away_team"].to_numpy()
    neutral = df["neutral"].to_numpy(dtype=bool)
    mask = np.array([(h in known) and (a in known) for h, a in zip(home, away)])

    pred_home = np.full(len(df), np.nan)
    pred_away = np.full(len(df), np.nan)
    if mask.any():
        h, a, n = home[mask], away[mask], neutral[mask]
        pred_home[mask] = (
            ratings.league_mean
            + off.loc[h].to_numpy()
            + def_.loc[a].to_numpy()
            + 0.5 * ratings.hfa * (~n).astype(float)
        )
        pred_away[mask] = (
            ratings.league_mean
            + off.loc[a].to_numpy()
            + def_.loc[h].to_numpy()
            - 0.5 * ratings.hfa * (~n).astype(float)
        )
    return pred_home, pred_away, mask


def _ratings_predictions(ratings: RatingsResult, train: pd.DataFrame, test: pd.DataFrame) -> Predictions:
    # Train residuals → sigma
    tr_h, tr_a, tr_mask = _ratings_score_preds(ratings, train)
    tr_pred_margin = tr_h - tr_a
    tr_actual_margin = (train["home_score"] - train["away_score"]).to_numpy(dtype=float)
    sigma = float(np.std(tr_pred_margin[tr_mask] - tr_actual_margin[tr_mask], ddof=1))

    h, a, mask = _ratings_score_preds(ratings, test)
    return Predictions(
        pred_margin=h - a,
        pred_home=h,
        pred_away=a,
        mask=mask,
        sigma=sigma,
    )


# ===================================================================
# Model registry
# ===================================================================

ModelFn = Callable[[pd.DataFrame, pd.DataFrame], Predictions]


def _margin_only_predictions(test: pd.DataFrame, pred_margin: np.ndarray, sigma: float) -> Predictions:
    """Helper for baselines that only predict margins, not per-side scores."""
    return Predictions(
        pred_margin=pred_margin,
        pred_home=np.full(len(test), np.nan),
        pred_away=np.full(len(test), np.nan),
        mask=np.ones(len(test), dtype=bool),
        sigma=sigma,
    )


def _baseline_tie(train: pd.DataFrame, test: pd.DataFrame) -> Predictions:
    train_actual = (train["home_score"] - train["away_score"]).to_numpy(dtype=float)
    sigma = float(np.std(train_actual, ddof=1))
    return _margin_only_predictions(test, np.zeros(len(test)), sigma)


def _baseline_hfa(train: pd.DataFrame, test: pd.DataFrame) -> Predictions:
    hfa = float(((train["home_score"] - train["away_score"])[~train["neutral"]]).mean())
    train_pred = np.where(train["neutral"].to_numpy(dtype=bool), 0.0, hfa)
    train_actual = (train["home_score"] - train["away_score"]).to_numpy(dtype=float)
    sigma = float(np.std(train_pred - train_actual, ddof=1))

    test_pred = np.where(test["neutral"].to_numpy(dtype=bool), 0.0, hfa)
    return _margin_only_predictions(test, test_pred, sigma)


def _baseline_avg_runs(train: pd.DataFrame, test: pd.DataFrame) -> Predictions:
    """Each team's predicted runs = season-average runs scored/allowed (no opponent
    adjustment) plus train-set HFA. Predicts per-side scores too."""
    long = pd.concat([
        train.rename(columns={"home_team": "team", "home_score": "rs", "away_score": "ra"})[["team", "rs", "ra"]],
        train.rename(columns={"away_team": "team", "away_score": "rs", "home_score": "ra"})[["team", "rs", "ra"]],
    ])
    avg = long.groupby("team").mean()
    hfa = float(((train["home_score"] - train["away_score"])[~train["neutral"]]).mean())

    def predict(df):
        home = df["home_team"].to_numpy()
        away = df["away_team"].to_numpy()
        neutral = df["neutral"].to_numpy(dtype=bool)
        mask = np.array([(h in avg.index) and (a in avg.index) for h, a in zip(home, away)])
        ph = np.full(len(df), np.nan)
        pa = np.full(len(df), np.nan)
        if mask.any():
            h, a, n = home[mask], away[mask], neutral[mask]
            ph[mask] = (avg.loc[h, "rs"].to_numpy() + avg.loc[a, "ra"].to_numpy()) / 2 + 0.5 * hfa * (~n).astype(float)
            pa[mask] = (avg.loc[a, "rs"].to_numpy() + avg.loc[h, "ra"].to_numpy()) / 2 - 0.5 * hfa * (~n).astype(float)
        return ph, pa, mask

    tr_h, tr_a, tr_mask = predict(train)
    tr_actual_margin = (train["home_score"] - train["away_score"]).to_numpy(dtype=float)
    sigma = float(np.std((tr_h - tr_a)[tr_mask] - tr_actual_margin[tr_mask], ddof=1))

    h, a, mask = predict(test)
    return Predictions(
        pred_margin=h - a,
        pred_home=h,
        pred_away=a,
        mask=mask,
        sigma=sigma,
    )


def _iter_mean(train, test, **kwargs):
    return _ratings_predictions(fit_ratings(train, **kwargs), train, test)


def _ridge(train, test, **kwargs):
    return _ratings_predictions(fit_ratings_ridge(train, **kwargs), train, test)


def _aridge(train, test, **kwargs):
    return _ratings_predictions(fit_ratings_adaptive_ridge(train, **kwargs), train, test)


# ---------- XGBoost ----------

def _rolling_form(games_sorted: pd.DataFrame, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For each game (in chronological order), return each team's mean runs scored
    and runs allowed in their previous ``n`` games (NaN if no history yet).
    Strict look-back: a game never sees itself or future games.
    """
    from collections import deque
    n_games = len(games_sorted)
    h_rs = np.full(n_games, np.nan)
    h_ra = np.full(n_games, np.nan)
    a_rs = np.full(n_games, np.nan)
    a_ra = np.full(n_games, np.nan)

    home = games_sorted["home_team"].to_numpy()
    away = games_sorted["away_team"].to_numpy()
    hs = games_sorted["home_score"].to_numpy()
    asc = games_sorted["away_score"].to_numpy()
    history: dict[str, deque] = {}

    for i in range(n_games):
        for team, rs_arr, ra_arr in (
            (home[i], h_rs, h_ra),
            (away[i], a_rs, a_ra),
        ):
            hist = history.get(team)
            if hist:
                rs_arr[i] = sum(x[0] for x in hist) / len(hist)
                ra_arr[i] = sum(x[1] for x in hist) / len(hist)
        history.setdefault(home[i], deque(maxlen=n)).append((hs[i], asc[i]))
        history.setdefault(away[i], deque(maxlen=n)).append((asc[i], hs[i]))

    return h_rs, h_ra, a_rs, a_ra


def _build_xgb_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    ratings: RatingsResult,
    form_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X_train, mask_train, X_test, mask_test) aligned to original row order."""
    full = (
        pd.concat([
            train.assign(_split="train", _idx=np.arange(len(train))),
            test.assign(_split="test", _idx=np.arange(len(test))),
        ])
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )
    h_rs, h_ra, a_rs, a_ra = _rolling_form(full, n=form_n)
    full["_h_rs"] = h_rs; full["_h_ra"] = h_ra
    full["_a_rs"] = a_rs; full["_a_ra"] = a_ra

    known = set(ratings.off.index)
    full["_mask"] = full["home_team"].isin(known) & full["away_team"].isin(known)
    full["_off_h"] = full["home_team"].map(ratings.off)
    full["_def_h"] = full["home_team"].map(ratings.def_)
    full["_off_a"] = full["away_team"].map(ratings.off)
    full["_def_a"] = full["away_team"].map(ratings.def_)
    full["_neutral"] = full["neutral"].astype(float)

    feat_cols = ["_off_h", "_def_h", "_off_a", "_def_a", "_neutral",
                 "_h_rs", "_h_ra", "_a_rs", "_a_ra"]

    tr = full[full["_split"] == "train"].sort_values("_idx")
    te = full[full["_split"] == "test"].sort_values("_idx")
    return (
        tr[feat_cols].to_numpy(dtype=np.float64),
        tr["_mask"].to_numpy(dtype=bool),
        te[feat_cols].to_numpy(dtype=np.float64),
        te["_mask"].to_numpy(dtype=bool),
    )


def _xgb(train, test, form_n: int = 10, **xgb_params) -> Predictions:
    """Fit a base ridge model, build features, train two XGB regressors
    (home_score and away_score), report Predictions."""
    import xgboost as xgb

    base = fit_ratings_ridge(train, lam=3, recency_half_life=45)
    X_tr, mask_tr, X_te, mask_te = _build_xgb_features(train, test, base, form_n)

    y_h = train["home_score"].to_numpy(dtype=float)
    y_a = train["away_score"].to_numpy(dtype=float)

    params = dict(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        objective="reg:squarederror",
        verbosity=0,
    )
    params.update(xgb_params)

    m_h = xgb.XGBRegressor(**params)
    m_a = xgb.XGBRegressor(**params)
    m_h.fit(X_tr[mask_tr], y_h[mask_tr])
    m_a.fit(X_tr[mask_tr], y_a[mask_tr])

    # Sigma from training residuals
    pred_h_tr = m_h.predict(X_tr[mask_tr])
    pred_a_tr = m_a.predict(X_tr[mask_tr])
    actual_margin_tr = y_h[mask_tr] - y_a[mask_tr]
    sigma = float(np.std((pred_h_tr - pred_a_tr) - actual_margin_tr, ddof=1))

    pred_h_te = np.full(len(test), np.nan)
    pred_a_te = np.full(len(test), np.nan)
    if mask_te.any():
        pred_h_te[mask_te] = m_h.predict(X_te[mask_te])
        pred_a_te[mask_te] = m_a.predict(X_te[mask_te])

    return Predictions(
        pred_margin=pred_h_te - pred_a_te,
        pred_home=pred_h_te,
        pred_away=pred_a_te,
        mask=mask_te,
        sigma=sigma,
    )


MODELS: dict[str, ModelFn] = {
    "baseline:tie":          _baseline_tie,
    "baseline:hfa":          _baseline_hfa,
    "baseline:avg_runs":     _baseline_avg_runs,
    "iter_mean":             _iter_mean,
    "iter_mean:blowout=8":   lambda tr, te: _iter_mean(tr, te, blowout_cap=8),
    "ridge:lam=1":           lambda tr, te: _ridge(tr, te, lam=1),
    "ridge:lam=3":           lambda tr, te: _ridge(tr, te, lam=3),
    "ridge:lam=5":           lambda tr, te: _ridge(tr, te, lam=5),
    "ridge:lam=10":          lambda tr, te: _ridge(tr, te, lam=10),
    "ridge:lam=20":          lambda tr, te: _ridge(tr, te, lam=20),
    "ridge:lam=3+blowout=8": lambda tr, te: _ridge(tr, te, lam=3, blowout_cap=8),
    "aridge:lam=1":          lambda tr, te: _aridge(tr, te, lam=1),
    "aridge:lam=3":          lambda tr, te: _aridge(tr, te, lam=3),
    "aridge:lam=5":          lambda tr, te: _aridge(tr, te, lam=5),
    "aridge:lam=10":         lambda tr, te: _aridge(tr, te, lam=10),
    "aridge:lam=3+blowout=8": lambda tr, te: _aridge(tr, te, lam=3, blowout_cap=8),
    # Recency-weighted ridge variants. Half-life is in days; the season is ~70 days
    # so half_life=30 strongly favors the last month, half_life=60 is gentler.
    "ridge:lam=3+hl=30":      lambda tr, te: _ridge(tr, te, lam=3, recency_half_life=30),
    "ridge:lam=3+hl=45":      lambda tr, te: _ridge(tr, te, lam=3, recency_half_life=45),
    "ridge:lam=3+hl=60":      lambda tr, te: _ridge(tr, te, lam=3, recency_half_life=60),
    "ridge:lam=3+hl=90":      lambda tr, te: _ridge(tr, te, lam=3, recency_half_life=90),
    "aridge:lam=10+hl=30":    lambda tr, te: _aridge(tr, te, lam=10, recency_half_life=30),
    "aridge:lam=10+hl=45":    lambda tr, te: _aridge(tr, te, lam=10, recency_half_life=45),
    "aridge:lam=10+hl=60":    lambda tr, te: _aridge(tr, te, lam=10, recency_half_life=60),
    "ridge:lam=3+hl=45+blowout=8": lambda tr, te: _ridge(tr, te, lam=3, recency_half_life=45, blowout_cap=8),
    "xgb:default":                 lambda tr, te: _xgb(tr, te),
    "xgb:shallow":                 lambda tr, te: _xgb(tr, te, max_depth=3, n_estimators=600),
    "xgb:deep":                    lambda tr, te: _xgb(tr, te, max_depth=6, n_estimators=300),
    "xgb:form=20":                 lambda tr, te: _xgb(tr, te, form_n=20),
}


# ===================================================================
# Runner + history persistence
# ===================================================================

def run_models(
    games: pd.DataFrame,
    n_test: int,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    train, test = chrono_split(games, n_test=n_test)
    names = model_names or list(MODELS.keys())
    rows = []
    ts = datetime.now().isoformat(timespec="seconds")
    for name in names:
        if name not in MODELS:
            raise KeyError(f"unknown model {name!r}; known: {sorted(MODELS)}")
        preds = MODELS[name](train, test)
        m = metrics_from_predictions(preds, test)
        rows.append({
            "timestamp": ts,
            "model": name,
            "n_test": n_test,
            "n_train": len(train),
            "n_rated": m.n,
            "MAE": round(m.margin_mae, 4),
            "RMSE": round(m.margin_rmse, 4),
            "bias": round(m.margin_bias, 4),
            "win_acc": round(m.win_acc, 4),
            "log_loss": round(m.log_loss, 4),
            "brier": round(m.brier, 4),
            "home_mae": round(m.home_mae, 4) if not np.isnan(m.home_mae) else None,
            "away_mae": round(m.away_mae, 4) if not np.isnan(m.away_mae) else None,
            "sigma": round(m.sigma, 4),
            "skipped": m.n_unrated,
        })
    return pd.DataFrame(rows)


def append_history(df: pd.DataFrame, path: Path = HISTORY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path, nrows=0)
        # If columns changed, start a new file alongside the old one rather than corrupt it.
        if list(existing.columns) != list(df.columns):
            backup = path.with_suffix(".csv.legacy")
            path.rename(backup)
            df.to_csv(path, index=False)
            return
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def load_history(path: Path = HISTORY_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["timestamp"])


def best_per_model(history: pd.DataFrame, by: str = "MAE") -> pd.DataFrame:
    if history.empty:
        return history
    higher_better = {"win_acc"}
    asc = by not in higher_better
    return (
        history.sort_values(by, ascending=asc)
        .groupby("model", as_index=False)
        .first()
        .sort_values(by, ascending=asc)
        .reset_index(drop=True)
    )


# ===================================================================
# CLI
# ===================================================================

DISPLAY_COLS = ["model", "MAE", "RMSE", "bias", "win_acc", "log_loss", "brier", "home_mae", "away_mae", "sigma"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rating models.")
    parser.add_argument("--n-test", type=int, default=400)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--history", action="store_true", help="show best historical run per model and exit")
    parser.add_argument("--sort", default="MAE", help="column to sort by (default MAE)")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for name in MODELS:
            print(name)
        return

    if args.history:
        h = load_history()
        if h.empty:
            print("no history yet")
            return
        print(f"Best historical run per model (by {args.sort}):")
        cols = [c for c in DISPLAY_COLS if c in h.columns] + ["n_test", "timestamp"]
        print(best_per_model(h, by=args.sort)[cols].to_string(index=False))
        return

    games = filter_to_core(fetch_raw())
    print(f"loaded {len(games)} core games, holding out last {args.n_test} as test set")
    df = run_models(games, n_test=args.n_test, model_names=args.models)

    higher_better = {"win_acc"}
    df_sorted = df.sort_values(args.sort, ascending=args.sort not in higher_better).reset_index(drop=True)
    print()
    print("This run:")
    print(df_sorted[DISPLAY_COLS].to_string(index=False))

    if not args.no_save:
        append_history(df)
        print(f"\nappended {len(df)} rows to {HISTORY_PATH}")


if __name__ == "__main__":
    main()
