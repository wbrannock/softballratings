# softballratings

**[View the live ratings →](https://wbrannock.github.io/softballratings/)**

Opponent-adjusted offensive and defensive ratings for D1 college softball, fit via ridge regression with recency weighting and updated daily.

## What it does

Every team gets two numbers: an **offensive rating** (runs scored above league average vs. a generic defense) and a **defensive rating** (runs allowed above league average vs. a generic offense). Both are *opponent-adjusted* — a team isn't credited for piling up runs against a soft schedule, and isn't penalized for losing tight games to top opponents. Combine the two and you get a **net rating**: the team's expected margin against a perfectly average opponent on a neutral field.

The model is fit jointly across every D1 game played this season. There's no preseason prior, no poll input, no human voting. It's a pure performance-on-the-field measurement.

## Quick example

| Rank | Team | Net | Off | Def |
|---:|---|---:|---:|---:|
| 1 | Oklahoma | 9.60 | +6.06 | −3.54 |
| 2 | Texas Tech | 8.63 | +4.86 | −3.78 |
| 3 | Arkansas | 8.31 | +4.47 | −3.85 |
| 4 | UCLA | 8.23 | +6.26 | −1.97 |
| 5 | Florida | 7.62 | +4.14 | −3.48 |
| … | … | … | … | … |

A net rating of 9.60 means *Oklahoma is expected to beat an average D1 team by about 9.6 runs on a neutral field*. Any matchup can be converted into a predicted final score and a win probability with one function call.

## Use it locally

```bash
uv sync                                          # install deps
uv run python -m softballratings.rate            # fit and print the latest top 25
uv run python -m softballratings.rate --top 50   # show 50 instead
jupyter notebook notebooks/explore.ipynb         # browse all 308 teams interactively
```

The first run downloads and caches the season's games, then fits the model. Subsequent runs reuse the cache (auto-refreshing when it's more than 6 hours old). To force a fresh pull:

```bash
uv run python -m softballratings.rate --refresh
```

## Predicting a matchup

```python
from softballratings.rate import build_ratings, predict_game

ratings = build_ratings()
predict_game(ratings, "Oklahoma", "Texas", neutral=False)
# {
#   'home': 'Oklahoma', 'away': 'Texas', 'neutral': False,
#   'pred_home_score': 7.5,
#   'pred_away_score': 5.6,
#   'pred_margin': 1.9,
#   'p_home_wins': 0.66,
# }
```

`p_home_wins` comes from `Φ(pred_margin / σ)`, where σ ≈ 4.6 is the model's residual standard deviation — roughly the irreducible single-game noise of softball as a sport.

## How to read a rating

| value | meaning |
|---|---|
| `off ≈ 0`, `def ≈ 0` | a perfectly average D1 team |
| `off = +3` | scores ~3 more runs than average vs. a generic defense |
| `def = −3` | allows ~3 fewer runs than average vs. a generic offense |
| `net ≈ +5` | beats an average team by ~5 runs on a neutral field |
| `net ≈ +9–10` | national title contender |
| `net ≈ −5` | bottom-quartile D1 |
| `p_beat_avg = 0.95` | wins 95% of single games against an average opponent |

**Sign convention:** `Off` positive = good offense (scores more). `Def` *negative* = good defense (allows fewer). Net = `Off − Def`, always larger-is-better.

## The model

For each game, two equations:

```
home_score = league_mean + Off[home] + Def[away] + 0.5·HFA·(1 − neutral)
away_score = league_mean + Off[away] + Def[home] − 0.5·HFA·(1 − neutral)
```

Stacked across every D1 game and solved as a single sparse ridge regression. Key ingredients:

- **Ridge penalty (λ = 3)** shrinks each team toward league average. Stabilizes thinly-evidenced teams and resolves a small identifiability degeneracy in the linear system.
- **Exponential recency weighting (45-day half-life)** so games from the last month count roughly twice as much as opening weekend.
- **Globally-fit home-field advantage** (~0.27 runs in the current data), shared across all teams.
- **Connectivity filter** that iteratively prunes teams with too few games against the surviving set, so opponents from non-D1 leagues don't pollute the ratings.

The hyperparameters were chosen with a held-out chronological test set. The production model has the best calibrated win probabilities (lowest log loss) of any variant tested and is within 0.01 runs of the best margin MAE. The eval harness lives in `softballratings/eval.py` if you want to try variations.

**Accuracy:** mean absolute error on predicted scoring differential is about **4.0 runs per game**, which is roughly 9% above the theoretical minimum for a model with our residual variance. Comparable in noise-relative terms to KenPom-class basketball ratings. The remaining error is largely irreducible single-game variance — softball is a 7-inning sport dominated by one pitcher per side, and variance is high.

## Live web page

`docs/index.html` is a single self-contained static page rendered from `data/ratings.csv` — sortable, mobile-friendly, with a team filter. A GitHub Action in `.github/workflows/daily.yml` re-runs the model every morning, regenerates the page, and commits both the CSV and the HTML. Hosted via GitHub Pages, no server, no monthly cost.

To enable on your own fork:

1. Make the repository public.
2. **Settings → Pages** → Source: *Deploy from a branch* → Branch: **main**, Folder: **/docs**.
3. **Settings → Actions → General** → *Workflow permissions* → **Read and write permissions**.
4. (optional) Trigger the first run from the Actions tab → "Daily ratings update" → "Run workflow".

After the first run, the page is live at:

```
https://<username>.github.io/<repo>/
```

The cron in the workflow runs at 13:00 UTC (≈ 9am ET); change it if you want a different schedule. Manual runs are always available from the Actions tab.

## Daily workflow (locally)

```bash
uv run python -m softballratings.rate          # auto-refresh + refit + show top 25 + movers
uv run python -m softballratings.rate --top 50
uv run python -m softballratings.rate --refresh
```

Each run prints the top-N table and the biggest rank movers since the previous run. The model output is also written to `data/ratings.csv` (and the prior run is rotated to `data/ratings_prev.csv`) so the next invocation can compute the diff.

To re-validate the model after changing something:

```bash
uv run python -m softballratings.eval
```

The eval harness holds out the last 400 games, fits every registered model variant on the rest, and reports MAE / RMSE / win accuracy / log loss / Brier / per-side score MAE. Results append to `data/eval_history.csv` so longitudinal comparisons are preserved across runs.

## Repository layout

```
softballratings/
  scrape.py    Game-result loader, parser, and connectivity-based filter
               that iteratively prunes weakly-connected teams.
  ratings.py   Three rating solvers, all fitting the same linear model:
                 - fit_ratings                iterative fixed-point
                 - fit_ratings_ridge          sparse ridge least-squares (production)
                 - fit_ratings_adaptive_ridge per-team penalty scaled by
                                              eigenvector centrality of the
                                              schedule graph
               All accept blowout_cap and recency_half_life knobs.
  rate.py      Production rating + prediction helpers, daily CLI with
               cache freshness check and movers diff.
  eval.py      Train/test eval harness with a model registry and persistent
               history. Add a new model to MODELS and rerun to score it
               against everything that came before.
  web.py       Renders data/ratings.csv to a self-contained static HTML page.
notebooks/
  explore.ipynb  Sanity checks, side-by-side model comparison, full
                 scrollable production rating table, sample matchup.
docs/
  index.html   The public web page (auto-generated, served by GitHub Pages).
data/
  ratings.csv       Latest production ratings (committed)
  ratings_prev.csv  One run ago, used for movers diff (committed)
  games.csv         Local game cache (gitignored)
  eval_history.csv  Longitudinal log of every eval run (gitignored)
```

## Adding a new model variant

1. Write a `(train, test) -> Predictions` function in `eval.py` (or build on the existing `_ridge` / `_aridge` / `_iter_mean` helpers for variants).
2. Drop it into the `MODELS` dict with a descriptive name.
3. Run `uv run python -m softballratings.eval` — your variant is scored alongside every existing model and the result is appended to history.
4. `uv run python -m softballratings.eval --history` shows the best-ever score per model across every run.
