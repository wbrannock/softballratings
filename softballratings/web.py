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
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" id="theme-color" content="#fbfbfc">
<title>D1 Softball Ratings</title>
<link rel="preconnect" href="https://rsms.me/">
<link rel="stylesheet" href="https://rsms.me/inter/inter.css">
<script>
  (function() {{
    try {{
      var stored = localStorage.getItem('theme');
      var prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
      var theme = stored === 'light' || stored === 'dark' ? stored : (prefersDark ? 'dark' : 'light');
      document.documentElement.setAttribute('data-theme', theme);
      var meta = document.getElementById('theme-color');
      if (meta) meta.setAttribute('content', theme === 'dark' ? '#08090a' : '#fbfbfc');
    }} catch (e) {{}}
  }})();
</script>
<style>
  :root {{
    --bg: #fbfbfc;
    --bg-grad:
      radial-gradient(900px 500px at 50% -260px, rgba(94, 106, 210, 0.10) 0%, transparent 60%),
      radial-gradient(700px 400px at 90% -200px, rgba(112, 159, 255, 0.08) 0%, transparent 60%),
      #fbfbfc;
    --surface: #ffffff;
    --surface-2: #f6f7f8;
    --surface-3: #f0f1f3;
    --text: #08090a;
    --text-2: #3c3f44;
    --muted: #6f7177;
    --muted-2: #9ea1a8;
    --border: rgba(8, 9, 10, 0.08);
    --border-strong: rgba(8, 9, 10, 0.14);
    --pos: #2f7a4d;
    --neg: #c1432d;
    --accent: #5e6ad2;
    --accent-2: #7170ff;
    --accent-soft: rgba(94, 106, 210, 0.12);
    --row-hover: rgba(8, 9, 10, 0.035);
    --shadow-sm: 0 1px 0 rgba(8, 9, 10, 0.04), 0 1px 2px rgba(8, 9, 10, 0.04);
    --shadow-md:
      0 1px 0 rgba(8, 9, 10, 0.04),
      0 1px 2px rgba(8, 9, 10, 0.04),
      0 12px 32px -8px rgba(8, 9, 10, 0.08);
    --ring: 0 0 0 3px var(--accent-soft);
  }}
  :root[data-theme="dark"] {{
      --bg: #08090a;
      --bg-grad:
        radial-gradient(900px 500px at 50% -240px, rgba(112, 113, 255, 0.18) 0%, transparent 60%),
        radial-gradient(700px 400px at 90% -180px, rgba(94, 106, 210, 0.14) 0%, transparent 60%),
        #08090a;
      --surface: #101113;
      --surface-2: #131517;
      --surface-3: #18191c;
      --text: #f7f8f8;
      --text-2: #d0d3d8;
      --muted: #8a8f98;
      --muted-2: #62666d;
      --border: rgba(255, 255, 255, 0.08);
      --border-strong: rgba(255, 255, 255, 0.14);
      --pos: #4cb782;
      --neg: #eb5757;
      --accent: #8d95f2;
      --accent-2: #b1b5ff;
      --accent-soft: rgba(141, 149, 242, 0.18);
      --row-hover: rgba(255, 255, 255, 0.035);
      --shadow-sm: 0 1px 0 rgba(0, 0, 0, 0.4), 0 1px 2px rgba(0, 0, 0, 0.3);
      --shadow-md:
        0 1px 0 rgba(0, 0, 0, 0.4),
        0 1px 2px rgba(0, 0, 0, 0.3),
        0 16px 40px -8px rgba(0, 0, 0, 0.5);
      --ring: 0 0 0 3px var(--accent-soft);
  }}
  * {{ box-sizing: border-box; }}
  html {{ -webkit-text-size-adjust: 100%; }}
  body {{
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-feature-settings: "cv11", "ss01", "ss03";
    max-width: 1040px;
    margin: 0 auto;
    padding: 2.25rem 1.25rem 4rem;
    color: var(--text);
    background: var(--bg-grad);
    background-attachment: fixed;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    letter-spacing: -0.005em;
  }}
  @supports (font-variation-settings: normal) {{
    body {{ font-family: "Inter var", "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  }}
  header {{
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    margin-bottom: 1.5rem;
  }}
  .title-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
  }}
  h1 {{
    margin: 0;
    font-size: 1.75rem;
    font-weight: 600;
    letter-spacing: -0.022em;
    line-height: 1.15;
  }}
  .theme-toggle {{
    width: 2.1rem;
    height: 2.1rem;
    flex-shrink: 0;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text-2);
    border-radius: 8px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0;
    cursor: pointer;
    box-shadow: var(--shadow-sm);
    transition: color 120ms ease, border-color 120ms ease, background 120ms ease, transform 120ms ease;
  }}
  .theme-toggle:hover {{
    color: var(--text);
    border-color: var(--border-strong);
    background: var(--surface-2);
  }}
  .theme-toggle:active {{ transform: scale(0.96); }}
  .theme-toggle:focus-visible {{
    outline: none;
    border-color: var(--accent);
    box-shadow: var(--ring);
  }}
  .theme-toggle svg {{
    width: 1rem;
    height: 1rem;
    display: block;
  }}
  .theme-toggle .icon-moon {{ display: none; }}
  :root[data-theme="dark"] .theme-toggle .icon-sun {{ display: none; }}
  :root[data-theme="dark"] .theme-toggle .icon-moon {{ display: block; }}
  .meta {{ color: var(--muted); font-size: 0.875rem; letter-spacing: 0; }}
  .meta .dot {{ margin: 0 0.45rem; color: var(--muted-2); }}
  .stats {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.5rem;
    margin: 1.5rem 0 1.75rem;
  }}
  .stat {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.7rem 0.85rem;
    box-shadow: var(--shadow-sm);
  }}
  .stat .label {{
    font-size: 0.72rem;
    color: var(--muted);
    font-weight: 500;
    letter-spacing: 0;
  }}
  .stat .value {{
    display: block;
    font-size: 1.2rem;
    font-weight: 600;
    margin-top: 0.2rem;
    letter-spacing: -0.018em;
    color: var(--text);
  }}
  .controls {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 0 0 0.85rem;
  }}
  .search {{
    position: relative;
    flex: 1;
    min-width: 0;
  }}
  .search svg {{
    position: absolute;
    left: 0.8rem;
    top: 50%;
    transform: translateY(-50%);
    width: 0.95rem;
    height: 0.95rem;
    color: var(--muted-2);
    pointer-events: none;
  }}
  input[type="search"] {{
    width: 100%;
    padding: 0.55rem 2.4rem 0.55rem 2.2rem;
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 0.9rem;
    font-family: inherit;
    background: var(--surface);
    color: var(--text);
    box-shadow: var(--shadow-sm);
    transition: border-color 120ms ease, box-shadow 120ms ease, background 120ms ease;
    -webkit-appearance: none;
    appearance: none;
    letter-spacing: -0.003em;
  }}
  input[type="search"]::-webkit-search-decoration,
  input[type="search"]::-webkit-search-cancel-button {{ -webkit-appearance: none; }}
  input[type="search"]:hover {{ border-color: var(--border-strong); }}
  input[type="search"]:focus {{
    outline: none;
    border-color: var(--accent);
    box-shadow: var(--ring);
  }}
  input[type="search"]::placeholder {{ color: var(--muted-2); }}
  .clear-btn {{
    position: absolute;
    right: 0.4rem;
    top: 50%;
    transform: translateY(-50%);
    width: 1.4rem;
    height: 1.4rem;
    border: 0;
    background: transparent;
    color: var(--muted);
    font-size: 1rem;
    line-height: 1;
    cursor: pointer;
    border-radius: 6px;
    display: none;
    align-items: center;
    justify-content: center;
    padding: 0;
    transition: background 120ms ease, color 120ms ease;
  }}
  .clear-btn:hover {{ background: var(--surface-3); color: var(--text); }}
  .search.has-value .clear-btn {{ display: flex; }}
  .count {{
    color: var(--muted);
    font-size: 0.82rem;
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
  }}
  .table-wrap {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    box-shadow: var(--shadow-md);
    overflow: hidden;
    position: relative;
  }}
  .table-scroll {{
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }}
  table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-variant-numeric: tabular-nums;
  }}
  th, td {{
    padding: 0.55rem 0.85rem;
    text-align: left;
    white-space: nowrap;
  }}
  thead th {{
    position: sticky;
    top: 0;
    background: var(--surface-2);
    font-size: 0.72rem;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    font-weight: 500;
    letter-spacing: 0;
    z-index: 2;
    height: 2.25rem;
  }}
  th[aria-sort="ascending"],
  th[aria-sort="descending"] {{ color: var(--text); }}
  th[aria-sort="ascending"] .sort-indicator,
  th[aria-sort="descending"] .sort-indicator {{ color: var(--accent); }}
  td {{
    border-top: 1px solid var(--border);
    font-size: 0.875rem;
    color: var(--text-2);
  }}
  tbody tr:first-child td {{ border-top: none; }}
  tbody tr:hover td {{ background: var(--row-hover); }}
  tbody tr.hidden {{ display: none; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .rank {{
    width: 3rem;
    color: var(--muted);
    font-weight: 500;
    font-variant-numeric: tabular-nums;
  }}
  .sort-button {{
    width: 100%;
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0;
    border: 0;
    background: transparent;
    color: inherit;
    font: inherit;
    font-weight: 500;
    letter-spacing: 0;
    cursor: pointer;
    transition: color 120ms ease;
  }}
  th.num .sort-button {{ justify-content: flex-end; }}
  .sort-button:hover {{ color: var(--text); }}
  .sort-button:focus-visible {{
    outline: 2px solid var(--accent);
    outline-offset: 2px;
    border-radius: 4px;
  }}
  .sort-indicator {{
    min-width: 0.7rem;
    color: var(--muted-2);
    text-align: center;
    font-size: 0.7rem;
  }}
  .team {{
    font-weight: 500;
    color: var(--text);
    white-space: normal;
    min-width: 8rem;
  }}
  .pos {{ color: var(--pos); }}
  .neg {{ color: var(--neg); }}
  .net-cell strong {{ font-weight: 600; }}
  /* Sticky first two columns when horizontally scrolling */
  th.rank, td.rank,
  th.team-col, td.team-col {{
    position: sticky;
    background: var(--surface);
    z-index: 1;
  }}
  thead th.rank, thead th.team-col {{
    background: var(--surface-2);
    z-index: 3;
  }}
  th.rank, td.rank {{ left: 0; }}
  th.team-col, td.team-col {{ left: 3rem; }}
  tbody tr:hover td.rank,
  tbody tr:hover td.team-col {{ background: var(--surface-2); }}
  td.team-col, thead th.team-col {{
    box-shadow: inset -1px 0 0 var(--border);
  }}
  .empty {{
    padding: 2.25rem 1rem;
    text-align: center;
    color: var(--muted);
    font-size: 0.875rem;
  }}
  footer {{
    margin-top: 2rem;
    color: var(--muted);
    font-size: 0.82rem;
    text-align: center;
    line-height: 1.65;
  }}
  footer a {{
    color: var(--accent);
    text-decoration: none;
    transition: color 120ms ease;
  }}
  footer a:hover {{ color: var(--accent-2); text-decoration: underline; }}
  ::selection {{ background: var(--accent-soft); color: var(--text); }}
  @media (max-width: 720px) {{
    body {{ padding: 1.5rem 0.85rem 3rem; }}
    h1 {{ font-size: 1.45rem; }}
    .meta {{ font-size: 0.82rem; }}
    .stats {{
      grid-template-columns: repeat(2, 1fr);
      gap: 0.45rem;
      margin: 1.25rem 0 1.4rem;
    }}
    .stat {{ padding: 0.55rem 0.7rem; }}
    .stat .label {{ font-size: 0.68rem; }}
    .stat .value {{ font-size: 1.05rem; }}
    .controls {{ flex-wrap: wrap; }}
    .count {{ flex-basis: 100%; }}
    th, td {{ padding: 0.5rem 0.65rem; font-size: 0.82rem; }}
    .sort-button {{ font-size: 0.7rem; }}
    .rank {{ width: 2.5rem; }}
    th.team-col, td.team-col {{ left: 2.5rem; }}
    .team {{ min-width: 7rem; }}
  }}
  @media (max-width: 420px) {{
    body {{ padding: 1.25rem 0.6rem 2.5rem; }}
    h1 {{ font-size: 1.3rem; }}
    th, td {{ padding: 0.45rem 0.55rem; font-size: 0.8rem; }}
  }}
</style>
</head>
<body>
  <header>
    <div class="title-row">
      <h1>D1 Softball Ratings</h1>
      <button type="button" class="theme-toggle" id="theme-toggle" aria-label="Toggle dark mode" title="Toggle dark mode">
        <svg class="icon-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"></circle><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"></path></svg>
        <svg class="icon-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3a7 7 0 0 0 9.79 9.79z"></path></svg>
      </button>
    </div>
    <div class="meta">
      <span>Updated {updated}</span>
      <span class="dot">·</span>
      <span>Season {season_start} – {season_end}</span>
    </div>
  </header>

  <div class="stats">
    <div class="stat"><div class="label">Teams</div><span class="value">{n_teams}</span></div>
    <div class="stat"><div class="label">Games</div><span class="value">{n_games}</span></div>
    <div class="stat"><div class="label">League mean</div><span class="value">{league_mean:.2f} R/G</span></div>
    <div class="stat"><div class="label">Home advantage</div><span class="value">+{hfa:.2f} R</span></div>
    <div class="stat"><div class="label">Game noise σ</div><span class="value">{sigma:.2f} R</span></div>
  </div>

  <div class="controls">
    <div class="search" id="search">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="11" cy="11" r="7"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
      <input type="search" id="filter" placeholder="Filter teams…" autocomplete="off" inputmode="search" aria-label="Filter teams">
      <button type="button" class="clear-btn" id="clear" aria-label="Clear filter">×</button>
    </div>
    <div class="count" id="count">{n_teams} teams</div>
  </div>

  <div class="table-wrap">
    <div class="table-scroll">
      <table>
        <thead>
          <tr>
            <th class="rank" aria-sort="ascending"><button type="button" class="sort-button" data-sort-key="rank"><span>Rank</span><span class="sort-indicator" aria-hidden="true">▲</span></button></th>
            <th class="team-col" aria-sort="none"><button type="button" class="sort-button" data-sort-key="team"><span>Team</span><span class="sort-indicator" aria-hidden="true">↕</span></button></th>
            <th class="num" aria-sort="none"><button type="button" class="sort-button" data-sort-key="games"><span>G</span><span class="sort-indicator" aria-hidden="true">↕</span></button></th>
            <th class="num" aria-sort="none"><button type="button" class="sort-button" data-sort-key="net"><span>Net</span><span class="sort-indicator" aria-hidden="true">↕</span></button></th>
            <th class="num" aria-sort="none"><button type="button" class="sort-button" data-sort-key="off"><span>Off</span><span class="sort-indicator" aria-hidden="true">↕</span></button></th>
            <th class="num" aria-sort="none"><button type="button" class="sort-button" data-sort-key="def"><span>Def</span><span class="sort-indicator" aria-hidden="true">↕</span></button></th>
            <th class="num" aria-sort="none"><button type="button" class="sort-button" data-sort-key="p"><span>P(beat avg)</span><span class="sort-indicator" aria-hidden="true">↕</span></button></th>
          </tr>
        </thead>
        <tbody>
{rows}
        </tbody>
      </table>
    </div>
    <div class="empty" id="empty" hidden>No teams match your filter.</div>
  </div>

  <footer>
    KenPom-style opponent-adjusted ratings via ridge regression with recency weighting.<br>
    Higher Off = scores more vs. average. Lower (more negative) Def = allows fewer vs. average.
    Net = expected margin vs. an average team on a neutral field.<br>
    Methodology and code: <a href="{repo_url}">GitHub</a>
  </footer>

  <script>
    const themeToggle = document.getElementById('theme-toggle');
    const themeColorMeta = document.getElementById('theme-color');
    themeToggle.addEventListener('click', () => {{
      const current = document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
      const next = current === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      if (themeColorMeta) themeColorMeta.setAttribute('content', next === 'dark' ? '#08090a' : '#fbfbfc');
      try {{ localStorage.setItem('theme', next); }} catch (e) {{}}
    }});

    const input = document.getElementById('filter');
    const search = document.getElementById('search');
    const clearBtn = document.getElementById('clear');
    const tableBody = document.querySelector('tbody');
    const empty = document.getElementById('empty');
    const count = document.getElementById('count');
    const rows = Array.from(tableBody.querySelectorAll('tr'));
    const headers = Array.from(document.querySelectorAll('thead th'));
    const buttons = Array.from(document.querySelectorAll('.sort-button'));
    const totalTeams = rows.length;
    let sortState = {{ key: 'rank', direction: 'asc' }};

    const directionDefaults = {{
      rank: 'asc',
      team: 'asc',
      games: 'desc',
      net: 'desc',
      off: 'desc',
      def: 'asc',
      p: 'desc',
    }};

    function updateFilter() {{
      const q = input.value.trim().toLowerCase();
      search.classList.toggle('has-value', q.length > 0);
      let visible = 0;
      rows.forEach(row => {{
        const hide = q && !row.dataset.team.includes(q);
        row.classList.toggle('hidden', hide);
        if (!hide) visible++;
      }});
      empty.hidden = visible !== 0;
      count.textContent = q
        ? `${{visible}} of ${{totalTeams}}`
        : `${{totalTeams}} teams`;
    }}

    function setHeaderState() {{
      headers.forEach(header => {{
        const button = header.querySelector('.sort-button');
        if (!button) return;
        const active = button.dataset.sortKey === sortState.key;
        const indicator = button.querySelector('.sort-indicator');
        header.setAttribute(
          'aria-sort',
          active ? (sortState.direction === 'asc' ? 'ascending' : 'descending') : 'none',
        );
        indicator.textContent = active ? (sortState.direction === 'asc' ? '▲' : '▼') : '↕';
      }});
    }}

    function compareRows(left, right) {{
      const {{ key, direction }} = sortState;
      let result = 0;
      if (key === 'team') {{
        result = left.dataset.team.localeCompare(right.dataset.team);
      }} else {{
        result = Number.parseFloat(left.dataset[key]) - Number.parseFloat(right.dataset[key]);
      }}
      if (result === 0) {{
        result = Number.parseFloat(left.dataset.rank) - Number.parseFloat(right.dataset.rank);
      }}
      return direction === 'asc' ? result : -result;
    }}

    function applySort() {{
      rows.sort(compareRows);
      tableBody.append(...rows);
      updateFilter();
      setHeaderState();
    }}

    input.addEventListener('input', updateFilter);
    clearBtn.addEventListener('click', () => {{
      input.value = '';
      input.focus();
      updateFilter();
    }});

    buttons.forEach(button => {{
      button.addEventListener('click', () => {{
        const key = button.dataset.sortKey;
        if (sortState.key === key) {{
          sortState.direction = sortState.direction === 'asc' ? 'desc' : 'asc';
        }} else {{
          sortState = {{ key, direction: directionDefaults[key] }};
        }}
        applySort();
      }});
    }});

    setHeaderState();
    updateFilter();
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
        f'      <tr data-team="{team_esc.lower()}"'
        f' data-rank="{rank}"'
        f' data-games="{n}"'
        f' data-net="{net:.12f}"'
        f' data-off="{off:.12f}"'
        f' data-def="{def_:.12f}"'
        f' data-p="{p:.12f}">'
        f'<td class="rank">{rank}</td>'
        f'<td class="team team-col">{team_esc}</td>'
        f'<td class="num">{n}</td>'
        f'<td class="num net-cell {net_cls}"><strong>{net:+.2f}</strong></td>'
        f'<td class="num {off_cls}">{off:+.2f}</td>'
        f'<td class="num {def_cls}">{def_:+.2f}</td>'
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
