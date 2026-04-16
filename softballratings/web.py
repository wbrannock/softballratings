"""Render the current ratings into a self-contained static HTML page.

Reads ``data/ratings.csv`` (produced by ``softballratings.rate``) and writes
``docs/index.html`` for GitHub Pages.

Usage:
    uv run python -m softballratings.web
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import pandas as pd

from .rate import DEFAULT_RATINGS_PATH

DEFAULT_OUT = Path("docs/index.html")


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>D1 Softball Ratings</title>
<style>
  :root {{
    --bg: #fafafa;
    --card: #ffffff;
    --text: #1f2937;
    --muted: #6b7280;
    --border: #e5e7eb;
    --pos: #15803d;
    --neg: #b91c1c;
    --accent: #1d4ed8;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    max-width: 960px;
    margin: 2rem auto;
    padding: 0 1rem 3rem;
    color: var(--text);
    background: var(--bg);
    line-height: 1.5;
  }}
  h1 {{ margin: 0 0 0.25rem; font-size: 1.75rem; }}
  .meta {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 1.5rem; }}
  .stats {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 0.75rem;
    margin-bottom: 1.5rem;
  }}
  .stat {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem 1rem;
  }}
  .stat .label {{ font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }}
  .stat .value {{ display: block; font-size: 1.3rem; font-weight: 600; margin-top: 0.15rem; }}
  .controls {{ margin: 1rem 0 0.75rem; }}
  input[type="search"] {{
    width: 100%;
    padding: 0.6rem 0.85rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 0.95rem;
    background: var(--card);
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    font-variant-numeric: tabular-nums;
  }}
  th, td {{ padding: 0.55rem 0.85rem; text-align: left; }}
  th {{
    background: #f3f4f6;
    font-size: 0.78rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border-bottom: 1px solid var(--border);
  }}
  td {{ border-top: 1px solid var(--border); font-size: 0.93rem; }}
  tbody tr:first-child td {{ border-top: none; }}
  tbody tr:hover td {{ background: #f9fafb; }}
  .num {{ text-align: right; }}
  .rank {{ width: 3rem; color: var(--muted); }}
  .team {{ font-weight: 500; }}
  .pos {{ color: var(--pos); }}
  .neg {{ color: var(--neg); }}
  footer {{ margin-top: 2rem; color: var(--muted); font-size: 0.85rem; text-align: center; }}
  footer a {{ color: var(--accent); text-decoration: none; }}
  footer a:hover {{ text-decoration: underline; }}
  @media (max-width: 600px) {{
    body {{ margin: 1rem auto; }}
    h1 {{ font-size: 1.4rem; }}
    th, td {{ padding: 0.4rem 0.55rem; font-size: 0.85rem; }}
    .stat .value {{ font-size: 1.1rem; }}
  }}
</style>
</head>
<body>
  <h1>D1 Softball Ratings</h1>
  <div class="meta">Updated {updated} · Season {season_start} – {season_end}</div>

  <div class="stats">
    <div class="stat"><div class="label">Teams</div><span class="value">{n_teams}</span></div>
    <div class="stat"><div class="label">Games</div><span class="value">{n_games}</span></div>
    <div class="stat"><div class="label">League mean</div><span class="value">{league_mean:.2f} R/G</span></div>
    <div class="stat"><div class="label">Home advantage</div><span class="value">+{hfa:.2f} R</span></div>
    <div class="stat"><div class="label">Game noise σ</div><span class="value">{sigma:.2f} R</span></div>
  </div>

  <div class="controls">
    <input type="search" id="filter" placeholder="Filter teams…" autocomplete="off">
  </div>

  <table>
    <thead>
      <tr>
        <th class="rank">Rank</th>
        <th>Team</th>
        <th class="num">G</th>
        <th class="num">Off</th>
        <th class="num">Def</th>
        <th class="num">Net</th>
        <th class="num">P(beat avg)</th>
      </tr>
    </thead>
    <tbody>
{rows}
    </tbody>
  </table>

  <footer>
    KenPom-style opponent-adjusted ratings via ridge regression with recency weighting.<br>
    Higher Off = scores more vs. average. Lower (more negative) Def = allows fewer vs. average.
    Net = expected margin vs. an average team on a neutral field.<br>
    Source: <a href="https://masseyratings.com/csoft/ncaa-d1/ratings">Massey Ratings</a> ·
    Methodology and code: <a href="{repo_url}">GitHub</a>
  </footer>

  <script>
    const input = document.getElementById('filter');
    const rows = Array.from(document.querySelectorAll('tbody tr'));
    input.addEventListener('input', e => {{
      const q = e.target.value.trim().toLowerCase();
      rows.forEach(r => {{
        r.style.display = q && !r.dataset.team.includes(q) ? 'none' : '';
      }});
    }});
  </script>
</body>
</html>
"""


def _row(rank: int, team: str, n: int, off: float, def_: float, net: float, p: float) -> str:
    off_cls = "pos" if off > 0 else "neg" if off < 0 else ""
    def_cls = "pos" if def_ < 0 else "neg" if def_ > 0 else ""  # negative def is GOOD
    net_cls = "pos" if net > 0 else "neg" if net < 0 else ""
    team_esc = escape(team)
    return (
        f'      <tr data-team="{team_esc.lower()}">'
        f'<td class="rank">{rank}</td>'
        f'<td class="team">{team_esc}</td>'
        f'<td class="num">{n}</td>'
        f'<td class="num {off_cls}">{off:+.2f}</td>'
        f'<td class="num {def_cls}">{def_:+.2f}</td>'
        f'<td class="num {net_cls}"><strong>{net:+.2f}</strong></td>'
        f'<td class="num">{p:.1%}</td>'
        f'</tr>'
    )


def render_html(
    ratings_path: Path = DEFAULT_RATINGS_PATH,
    out_path: Path = DEFAULT_OUT,
    games_path: Path | None = Path("data/games.csv"),
    repo_url: str = "https://github.com/",
) -> Path:
    df = pd.read_csv(ratings_path)
    if df.empty:
        raise ValueError(f"{ratings_path} is empty — run softballratings.rate first")

    rows_html = "\n".join(
        _row(int(r["rank"]), r["team"], int(r["n_games"]),
             float(r["off"]), float(r["def"]), float(r["net"]),
             float(r["p_beat_avg"]))
        for _, r in df.iterrows()
    )

    season_start = season_end = "—"
    n_games = "—"
    if games_path is not None and games_path.exists():
        games = pd.read_csv(games_path, parse_dates=["date"])
        if not games.empty:
            season_start = games["date"].min().date().isoformat()
            season_end = games["date"].max().date().isoformat()
            n_games = f"{len(games):,}"

    page = PAGE.format(
        updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        season_start=season_start,
        season_end=season_end,
        n_teams=len(df),
        n_games=n_games,
        league_mean=float(df["league_mean"].iloc[0]),
        hfa=float(df["hfa"].iloc[0]),
        sigma=float(df["sigma"].iloc[0]),
        rows=rows_html,
        repo_url=repo_url,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render ratings to a static HTML page.")
    parser.add_argument("--ratings", default=str(DEFAULT_RATINGS_PATH))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--repo-url",
        default="https://github.com/",
        help="link shown in the page footer; set to your repo URL",
    )
    args = parser.parse_args()

    out = render_html(
        ratings_path=Path(args.ratings),
        out_path=Path(args.out),
        repo_url=args.repo_url,
    )
    print(f"wrote {out}  ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
