# softballratings

KenPom-style opponent-adjusted offensive and defensive ratings for D1 college softball, scraped from [Massey Ratings](https://masseyratings.com/scores.php?s=658934&sub=11590&all=1) and fit via ridge regression with recency weighting.

## Quickstart

```bash
uv sync                                            # install deps
uv run python -m softballratings.rate --refresh    # scrape, fit, write data/ratings.csv, print top 25
uv run python -m softballratings.eval              # run all model variants on a held-out test set
jupyter notebook notebooks/explore.ipynb           # browse all 308 teams interactively
```

## What's in the box

```
softballratings/
  scrape.py    Fetches Massey scores via curl_cffi (Cloudflare-aware), parses
               the <pre> block, normalizes home/away/neutral, caches to
               data/games.csv. Provides filter_to_core() to iteratively prune
               teams with fewer than N games against the surviving set —
               kills D2/NAIA contamination from the &all=1 endpoint.
  ratings.py   Three rating solvers, all using the same model:
                 home_score = mu + Off[h] + Def[a] + 0.5*HFA*(1-neutral)
                 away_score = mu + Off[a] + Def[h] - 0.5*HFA*(1-neutral)
               * fit_ratings              — iterative fixed-point
               * fit_ratings_ridge        — single sparse ridge least-squares
               * fit_ratings_adaptive_ridge — per-team penalty scaled by
                                              eigenvector centrality on the
                                              schedule graph
               All accept blowout_cap and recency_half_life knobs.
  eval.py     Train/test eval harness with a model registry, persistent
              history (data/eval_history.csv), and CLI. Reports MAE, RMSE,
              bias, win_acc, log_loss, Brier, per-side score MAE, and the
              implied noise sigma. Add a new model by dropping a callable
              into MODELS and rerunning — old runs stay queryable.
  rate.py     Production: build_ratings() fits the chosen model on the full
              dataset and returns/saves a tidy DataFrame with rank, off, def,
              net, and p_beat_avg. predict_game() does head-to-head matchup
              prediction with implied win probability.
notebooks/
  explore.ipynb  Sanity checks, side-by-side model comparison, full
                 scrollable production rating table, sample matchup.
data/                                       (gitignored)
  games.csv         scraper cache
  ratings.csv       latest production ratings
  eval_history.csv  longitudinal log of every model evaluation
```

## The model

For each game, two equations:

```
home_score = league_mean + Off[home] + Def[away] + 0.5 * HFA * (1 - neutral)
away_score = league_mean + Off[away] + Def[home] - 0.5 * HFA * (1 - neutral)
```

- `Off[t]` = team t's runs scored above league average vs. an average defense (positive = good offense).
- `Def[t]` = team t's runs allowed above league average vs. an average offense (negative = good defense).
- `HFA` = full home advantage in runs, fit globally from non-neutral games.
- `Net = Off - Def` = expected margin against a league-average team on a neutral field.

Production parameters (`PROD_LAM`, `PROD_HALF_LIFE` in `rate.py`):

- **ridge λ = 3** — shrinks each team rating toward zero by ~3 "phantom games" of regularization. Resolves the Off/Def shift degeneracy and dampens small-sample noise.
- **recency half-life = 45 days** — every game's contribution to the fit decays by half every 45 days. Late-season games count ~2× as much as opening weekend.
- **no blowout cap by default** — adding `blowout_cap=8` improves win accuracy slightly at a small MAE cost; opt in via `build_ratings(..., blowout_cap=8)`.

## How the model was chosen

The eval harness (`softballratings.eval`) holds out the last 400 games chronologically as a test set, fits each model variant on the remainder, and scores them on margin MAE/RMSE, win accuracy, log loss, Brier, and per-side score MAE. Every run appends to `data/eval_history.csv`, so adding a new model to the `MODELS` registry and rerunning produces a directly comparable scoreline.

What we tried, in rough order:

1. **Buggy iterative fit + no filtering** — opponent-adjusted but contaminated by D2/NAIA opponents from Massey's `&all=1` page. Top-25 had Henderson St, MI Dearborn, Wiley. Eval was barely better than a naive averages baseline.
2. **Connectivity filter (`filter_to_core`)** — iteratively dropped teams with <15 games against the surviving set. Cleaned the obvious noise but mid-major inflation remained.
3. **Model rewrite + ridge** — fixed a sign bug in the iterative solver and added `fit_ratings_ridge`. Both now agree on the same model. MAE dropped from 4.39 → 4.01, win accuracy 64% → 68%.
4. **Adaptive ridge with eigenvector centrality** — gave weakly-connected teams (HBCU/MEAC island clusters) a larger penalty. Improved win accuracy on harder games but didn't move MAE.
5. **Recency weighting** — exponential decay with half-life 45 days. Improved log loss / Brier marginally and gave the new best win accuracy when stacked with blowout cap (70.0%).
6. **XGBoost benchmark** — given pre-fit ridge ratings, rolling 10-game form, and a neutral flag. Tied the ridge family on MAE, **lost** on log loss (overfit train residuals), tied on win accuracy. Confirmed the ridge model captures essentially all the predictable signal — remaining ~4 runs of MAE is intrinsic single-game noise (sigma ≈ 4.6).

The final pick — `ridge:lam=3, half_life=45` — is within 0.01 runs of the best MAE, ties for best log loss, and is the simplest fast solver in the registry.

## Interpreting a rating

| value | meaning |
|---|---|
| `off ≈ 0`, `def ≈ 0` | a perfectly average D1 team |
| `off = +3` | scores ~3 more runs than average vs. a generic defense |
| `def = −3` | allows ~3 fewer runs than average vs. a generic offense |
| `net ≈ +5` | beats an average team by ~5 runs on a neutral field |
| `net ≈ +9–10` | national title contender |
| `net ≈ −5` | bottom-quartile D1 |
| `p_beat_avg = 0.95` | wins 95% of single games against an average opponent |

Predicting a head-to-head:

```python
from softballratings.rate import build_ratings, predict_game

ratings = build_ratings()
predict_game(ratings, "Oklahoma", "Texas", neutral=False)
# {'home': 'Oklahoma', 'away': 'Texas', 'neutral': False,
#  'pred_home_score': 7.5, 'pred_away_score': 5.6,
#  'pred_margin': 1.9, 'p_home_wins': 0.66}
```

`p_home_wins` comes from `Φ(pred_margin / sigma)` where σ is the model's training-residual std — the noise floor of softball as a sport, currently ~4.6 runs per game. Big margins compress to confident probabilities; small margins stay near 50/50.

## Daily refresh workflow

Massey is updated daily during the season. The default `rate` command auto-refreshes the scrape cache when it's more than 6 hours old, so a daily check is just one line:

```bash
uv run python -m softballratings.rate                # auto-refresh + refit + show top 25 + movers
uv run python -m softballratings.rate --top 50       # show more teams
uv run python -m softballratings.rate --refresh      # force re-scrape now
uv run python -m softballratings.rate --max-age 0    # same as --refresh
uv run python -m softballratings.rate --no-save      # dry run, don't touch CSVs
```

Each run:
1. Auto-refreshes the Massey scrape if `data/games.csv` is older than `--max-age` hours (default 6).
2. Reads the previous `data/ratings.csv` so it can compute movers.
3. Fits the production model on the full filtered dataset.
4. Rotates the previous file to `data/ratings_prev.csv` and writes the new ratings to `data/ratings.csv`.
5. Prints the top-N table plus the biggest rank movers since last run.

Sample output:

```
fit on 6123 games, 308 teams (2026-02-05 → 2026-04-15)
HFA = 0.265 runs    sigma = 4.612    league_mean = 5.141

Top 25:
 rank          team  n_games   off    def   net  p_beat_avg
    1      Oklahoma       44 6.056 -3.541 9.597       0.981
    2    Texas Tech       45 4.855 -3.778 8.633       0.969
    ...

Biggest movers since last run:
  ↑ Oregon            rank  19 → 15   (+4, net +0.31)
  ↓ Arizona           rank  22 → 28   (-6, net -0.42)
  ...

wrote data/ratings.csv (previous → data/ratings_prev.csv)
```

To re-validate the model against a held-out tail when you want to confirm nothing has drifted:

```bash
uv run python -m softballratings.eval
```

The scraper uses `curl_cffi` with Chrome TLS impersonation to bypass Cloudflare's JS challenge, so no manual page saving is required.

## Public web page (GitHub Pages + Actions)

The repo ships with a daily GitHub Action (`.github/workflows/daily.yml`) that scrapes Massey, refits the ratings, renders a static HTML page to `docs/index.html`, and commits the result. Combined with GitHub Pages, this gives you a public dashboard that auto-updates every morning with **zero hosting cost and no server**.

### One-time setup

1. **Make the repo public** (Settings → General → Danger Zone → Change visibility).
2. **Enable GitHub Pages**: Settings → Pages → "Build and deployment" → Source: **Deploy from a branch** → Branch: **main**, Folder: **/docs**. Save.
3. **Allow Actions to push commits**: Settings → Actions → General → "Workflow permissions" → **Read and write permissions**. Save.
4. (optional) Trigger the first run manually: Actions tab → "Daily ratings update" → "Run workflow".

After the first run, your page is live at:

```
https://<your-github-username>.github.io/<repo-name>/
```

### How the daily action works

```yaml
schedule: cron "0 13 * * *"   # 13:00 UTC = 9am ET
```

Each run:
1. `uv sync --frozen` — install the locked dependency set
2. `uv run python -m softballratings.rate --refresh` — re-scrape Massey, refit, write `data/ratings.csv` and `data/ratings_prev.csv`
3. `uv run python -m softballratings.web --repo-url …` — render `docs/index.html` from the CSV
4. Commit the three files (only if anything actually changed) and push

Adjust the cron in `.github/workflows/daily.yml` if you want a different time, or trigger manually any time from the Actions tab.

### Local preview

```bash
uv run python -m softballratings.rate         # generates data/ratings.csv
uv run python -m softballratings.web          # renders docs/index.html
open docs/index.html                          # macOS — view in browser
```

The page is a single self-contained HTML file with inline CSS and a tiny vanilla-JS team filter. No build step, no dependencies, no JavaScript libraries pulled from CDNs.

## Adding a new model variant

1. Write a `(train, test) -> Predictions` function in `eval.py` (or build on `_ridge` / `_aridge` / `_iter_mean` for variants).
2. Drop it into the `MODELS` dict with a descriptive name.
3. Rerun `uv run python -m softballratings.eval` — the new model is scored alongside every existing variant, and the result is appended to `data/eval_history.csv`.
4. `uv run python -m softballratings.eval --history` shows the best-ever score per model across all runs.
