from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh, lsqr


@dataclass
class RatingsResult:
    off: pd.Series       # opponent-adjusted runs scored above league mean
    def_: pd.Series      # opponent-adjusted runs allowed above league mean
    hfa: float           # full home-margin advantage in runs
    league_mean: float   # league-average runs per team-game
    n_iter: int
    converged: bool


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
# Model (used by both solvers, so they're directly comparable):
#
#   home_score = league_mean + Off[home] + Def[away] + 0.5*HFA*(1-neutral)
#   away_score = league_mean + Off[away] + Def[home] - 0.5*HFA*(1-neutral)
#
# Sign convention: Off positive means a team scores above league average.
# Def positive means a team *allows* above league average (i.e. WORSE defense).
# Net rating = Off - Def (good offense, low giveup).


def _blowout_weights(margins: np.ndarray, cap: float | None, k: float) -> np.ndarray:
    if cap is None:
        return np.ones_like(margins, dtype=float)
    excess = np.maximum(0.0, np.abs(margins) - cap)
    return k / (k + excess)


def _time_weights(games: pd.DataFrame, half_life_days: float | None) -> np.ndarray | None:
    """Exponential decay weights anchored to the latest game in the set.

    weight = 0.5 ** (age_days / half_life_days). With half_life=30, a game
    one half-life old counts half, two half-lives old counts a quarter, etc.
    Anchoring to the train set's max date (rather than "today") keeps the
    weighting reproducible across runs.
    """
    if half_life_days is None:
        return None
    dates = pd.to_datetime(games["date"])
    max_date = dates.max()
    age_days = (max_date - dates).dt.days.to_numpy(dtype=float)
    return 0.5 ** (age_days / half_life_days)


def _combine_weights(games: pd.DataFrame, margin: np.ndarray,
                     blowout_cap: float | None, blowout_k: float,
                     recency_half_life: float | None) -> np.ndarray:
    w = _blowout_weights(margin, blowout_cap, blowout_k)
    tw = _time_weights(games, recency_half_life)
    if tw is not None:
        w = w * tw
    return w


def _prep(games: pd.DataFrame):
    teams = pd.Index(sorted(set(games["home_team"]) | set(games["away_team"])))
    t_idx = {t: i for i, t in enumerate(teams)}
    home_i = games["home_team"].map(t_idx).to_numpy()
    away_i = games["away_team"].map(t_idx).to_numpy()
    hs = games["home_score"].to_numpy(dtype=float)
    as_ = games["away_score"].to_numpy(dtype=float)
    is_neutral = games["neutral"].to_numpy(dtype=bool)
    return teams, home_i, away_i, hs, as_, is_neutral


# ---------------------------------------------------------------------------
# Iterative fixed-point solver (no shrinkage)
# ---------------------------------------------------------------------------

def fit_ratings(
    games: pd.DataFrame,
    blowout_cap: float | None = None,
    blowout_k: float = 4.0,
    recency_half_life: float | None = None,
    max_iter: int = 200,
    tol: float = 1e-5,
) -> RatingsResult:
    teams, home_i, away_i, hs, as_, is_neutral = _prep(games)
    n = len(teams)

    margin = hs - as_
    w = _combine_weights(games, margin, blowout_cap, blowout_k, recency_half_life)

    league_mean = float(np.average(np.concatenate([hs, as_]),
                                   weights=np.concatenate([w, w])))

    off = np.zeros(n)
    defn = np.zeros(n)
    hfa = 0.0

    converged = False
    n_iter = 0
    for it in range(1, max_iter + 1):
        n_iter = it
        hfa_home = np.where(is_neutral, 0.0, hfa / 2.0)
        hfa_away = -hfa_home

        # Solve each side of each game equation for the team in question:
        #   Off[home] = hs - mu - Def[away] - hfa_home
        #   Off[away] = as - mu - Def[home] - hfa_away
        #   Def[away] = hs - mu - Off[home] - hfa_home
        #   Def[home] = as - mu - Off[away] - hfa_away
        off_obs_home = hs - league_mean - defn[away_i] - hfa_home
        off_obs_away = as_ - league_mean - defn[home_i] - hfa_away
        def_obs_away = hs - league_mean - off[home_i] - hfa_home
        def_obs_home = as_ - league_mean - off[away_i] - hfa_away

        new_off = np.zeros(n)
        new_def = np.zeros(n)
        wsum = np.zeros(n)

        np.add.at(new_off, home_i, w * off_obs_home)
        np.add.at(new_off, away_i, w * off_obs_away)
        np.add.at(new_def, away_i, w * def_obs_away)
        np.add.at(new_def, home_i, w * def_obs_home)
        np.add.at(wsum, home_i, w)
        np.add.at(wsum, away_i, w)

        new_off = np.where(wsum > 0, new_off / wsum, 0.0)
        new_def = np.where(wsum > 0, new_def / wsum, 0.0)

        # Recenter so league-average team has Off=Def=0.
        new_off -= new_off.mean()
        new_def -= new_def.mean()

        # Update HFA from non-neutral games.
        nn = ~is_neutral
        if nn.any():
            # Predicted home margin = (Off[h] - Off[a]) + (Def[a] - Def[h]) + HFA
            predicted_margin = (
                (new_off[home_i] - new_off[away_i])
                + (new_def[away_i] - new_def[home_i])
            )
            residual = margin[nn] - predicted_margin[nn]
            new_hfa = float(np.average(residual, weights=w[nn]))
        else:
            new_hfa = hfa

        delta = max(
            np.abs(new_off - off).max(),
            np.abs(new_def - defn).max(),
            abs(new_hfa - hfa),
        )
        off, defn, hfa = new_off, new_def, new_hfa
        if delta < tol:
            converged = True
            break

    return RatingsResult(
        off=pd.Series(off, index=teams, name="off"),
        def_=pd.Series(defn, index=teams, name="def"),
        hfa=hfa,
        league_mean=league_mean,
        n_iter=n_iter,
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Ridge least-squares solver
# ---------------------------------------------------------------------------

def _connectivity(home_i: np.ndarray, away_i: np.ndarray, n_teams: int) -> np.ndarray:
    """Eigenvector centrality of each team in the (undirected) schedule graph.

    A[i,j] = number of games between teams i and j. The dominant eigenvector
    of A scores each team by "you're central if you play teams that are
    themselves central" — recursively. Returns values normalized so the mean
    is 1.0, so they slot directly into ``lam_per_team = lam / centrality``.
    """
    # Build symmetric adjacency: each game contributes +1 to A[h,a] and A[a,h].
    rows = np.concatenate([home_i, away_i])
    cols = np.concatenate([away_i, home_i])
    data = np.ones(len(rows), dtype=np.float64)
    A = coo_matrix((data, (rows, cols)), shape=(n_teams, n_teams)).tocsr()

    # Largest-magnitude eigenvalue/vector. eigsh returns sign-arbitrary; abs it.
    _, vecs = eigsh(A, k=1, which="LA")
    c = np.abs(vecs[:, 0])
    c = c / c.mean()
    return c


def fit_ratings_adaptive_ridge(
    games: pd.DataFrame,
    lam: float = 3.0,
    blowout_cap: float | None = None,
    blowout_k: float = 4.0,
    recency_half_life: float | None = None,
    clip: tuple[float, float] = (0.5, 5.0),
) -> RatingsResult:
    """Ridge with per-team penalty strength scaled by schedule connectivity.

    For each team t, the ridge λ is scaled by ``median_connectivity / c_t``,
    where c_t is the mean game-count of t's opponents. Teams whose opponents
    are themselves sparsely connected get a larger penalty (pulled harder
    toward zero); well-connected teams keep close to the base ``lam``. The
    scale factor is clipped to ``clip`` to prevent extreme values.

    HFA is essentially unpenalized (one global scalar with plenty of evidence).
    """
    teams, home_i, away_i, hs, as_, is_neutral = _prep(games)
    n_teams = len(teams)
    n_games = len(games)

    margin = hs - as_
    w_game = _combine_weights(games, margin, blowout_cap, blowout_k, recency_half_life)
    league_mean = float(np.average(np.concatenate([hs, as_]),
                                   weights=np.concatenate([w_game, w_game])))

    # ----- per-team penalty -----
    # Centrality is normalized to mean 1, so dividing by it preserves the
    # league-wide effective lambda while reallocating across teams.
    c = _connectivity(home_i, away_i, n_teams)
    scale = np.clip(1.0 / np.maximum(c, 1e-6), clip[0], clip[1])
    lam_team = lam * scale
    lam_per_var = np.concatenate([lam_team, lam_team, [1e-6]])  # Off, Def, HFA
    sqrt_lam_per_var = np.sqrt(lam_per_var)

    # ----- design matrix (same shape as ridge, plus n_vars penalty rows) -----
    off_off = 0
    def_off = n_teams
    hfa_idx = 2 * n_teams
    n_vars = 2 * n_teams + 1

    nn = (~is_neutral).astype(float)
    n_rows_data = 2 * n_games

    rows = np.empty(3 * n_rows_data, dtype=np.int64)
    cols = np.empty(3 * n_rows_data, dtype=np.int64)
    vals = np.empty(3 * n_rows_data, dtype=np.float64)
    y = np.empty(n_rows_data, dtype=np.float64)

    g = np.arange(n_games)
    r_home = 2 * g
    r_away = 2 * g + 1

    base = 3 * r_home
    rows[base + 0] = r_home; cols[base + 0] = off_off + home_i; vals[base + 0] = 1.0
    rows[base + 1] = r_home; cols[base + 1] = def_off + away_i; vals[base + 1] = 1.0
    rows[base + 2] = r_home; cols[base + 2] = hfa_idx;          vals[base + 2] = 0.5 * nn

    base = 3 * r_away
    rows[base + 0] = r_away; cols[base + 0] = off_off + away_i; vals[base + 0] = 1.0
    rows[base + 1] = r_away; cols[base + 1] = def_off + home_i; vals[base + 1] = 1.0
    rows[base + 2] = r_away; cols[base + 2] = hfa_idx;          vals[base + 2] = -0.5 * nn

    y[r_home] = hs - league_mean
    y[r_away] = as_ - league_mean

    sqrt_w = np.sqrt(np.repeat(w_game, 2))
    vals = vals * sqrt_w[rows]
    y = y * sqrt_w

    # Append per-variable penalty rows.
    pen_rows = np.arange(n_vars) + n_rows_data
    pen_cols = np.arange(n_vars)
    pen_vals = sqrt_lam_per_var

    all_rows = np.concatenate([rows, pen_rows])
    all_cols = np.concatenate([cols, pen_cols])
    all_vals = np.concatenate([vals, pen_vals])
    y_full = np.concatenate([y, np.zeros(n_vars)])

    X = csr_matrix((all_vals, (all_rows, all_cols)), shape=(n_rows_data + n_vars, n_vars))
    res = lsqr(X, y_full, atol=1e-10, btol=1e-10, iter_lim=10000)
    beta = res[0]

    return RatingsResult(
        off=pd.Series(beta[off_off:off_off + n_teams], index=teams, name="off"),
        def_=pd.Series(beta[def_off:def_off + n_teams], index=teams, name="def"),
        hfa=float(beta[hfa_idx]),
        league_mean=league_mean,
        n_iter=int(res[2]),
        converged=res[1] in (1, 2),
    )


def fit_ratings_ridge(
    games: pd.DataFrame,
    lam: float = 3.0,
    blowout_cap: float | None = None,
    blowout_k: float = 4.0,
    recency_half_life: float | None = None,
) -> RatingsResult:
    """Solve the rating model as a single ridge regression.

    Two equations per game (home runs, away runs). Variables are
    [Off_0..Off_{n-1}, Def_0..Def_{n-1}, HFA].

    The ridge penalty ``lam`` shrinks every team rating toward zero and
    simultaneously resolves the Off↔Def shift degeneracy by picking the
    minimum-norm solution. ``lam`` is roughly in "phantom games" units —
    λ=0 is unregularized, λ≈3–10 is typical KenPom-style shrinkage.
    """
    teams, home_i, away_i, hs, as_, is_neutral = _prep(games)
    n_teams = len(teams)
    n_games = len(games)

    margin = hs - as_
    w_game = _combine_weights(games, margin, blowout_cap, blowout_k, recency_half_life)
    league_mean = float(np.average(np.concatenate([hs, as_]),
                                   weights=np.concatenate([w_game, w_game])))

    off_off = 0
    def_off = n_teams
    hfa_idx = 2 * n_teams
    n_vars = 2 * n_teams + 1

    nn = (~is_neutral).astype(float)

    # Three nonzeros per row: [Off, Def, HFA]
    n_rows = 2 * n_games
    rows = np.empty(3 * n_rows, dtype=np.int64)
    cols = np.empty(3 * n_rows, dtype=np.int64)
    vals = np.empty(3 * n_rows, dtype=np.float64)
    y = np.empty(n_rows, dtype=np.float64)

    g = np.arange(n_games)
    r_home = 2 * g       # row index for "home runs" equation
    r_away = 2 * g + 1   # row index for "away runs" equation

    # Home-runs eq: hs - mu = Off[home] + Def[away] + 0.5*HFA*nn
    base = 3 * r_home
    rows[base + 0] = r_home; cols[base + 0] = off_off + home_i; vals[base + 0] = 1.0
    rows[base + 1] = r_home; cols[base + 1] = def_off + away_i; vals[base + 1] = 1.0
    rows[base + 2] = r_home; cols[base + 2] = hfa_idx;          vals[base + 2] = 0.5 * nn

    # Away-runs eq: as - mu = Off[away] + Def[home] - 0.5*HFA*nn
    base = 3 * r_away
    rows[base + 0] = r_away; cols[base + 0] = off_off + away_i; vals[base + 0] = 1.0
    rows[base + 1] = r_away; cols[base + 1] = def_off + home_i; vals[base + 1] = 1.0
    rows[base + 2] = r_away; cols[base + 2] = hfa_idx;          vals[base + 2] = -0.5 * nn

    y[r_home] = hs - league_mean
    y[r_away] = as_ - league_mean

    # Per-row sqrt weights so lsqr minimizes weighted residuals.
    sqrt_w = np.sqrt(np.repeat(w_game, 2))  # interleaved (home, away, home, away, ...)
    # repeat order: w_game[0],w_game[0],w_game[1],w_game[1],... — but our row order is
    # 0,2,4,...(home rows) then 1,3,5...(away rows). repeat gives [w0,w0,w1,w1,...]
    # which lines up with rows 0,1,2,3,... so we can index by `rows`.
    vals = vals * sqrt_w[rows]
    y = y * sqrt_w

    X = csr_matrix((vals, (rows, cols)), shape=(n_rows, n_vars))

    res = lsqr(X, y, damp=np.sqrt(lam), atol=1e-10, btol=1e-10, iter_lim=10000)
    beta = res[0]

    return RatingsResult(
        off=pd.Series(beta[off_off:off_off + n_teams], index=teams, name="off"),
        def_=pd.Series(beta[def_off:def_off + n_teams], index=teams, name="def"),
        hfa=float(beta[hfa_idx]),
        league_mean=league_mean,
        n_iter=int(res[2]),
        converged=res[1] in (1, 2),
    )
