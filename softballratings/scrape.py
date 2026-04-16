from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DEFAULT_URL = "https://masseyratings.com/scores.php?s=658934&sub=11590&all=1"
D1_ONLY_URL = "https://masseyratings.com/scores.php?s=658934&sub=11590"
DEFAULT_CACHE = Path("data/games.csv")
D1_TEAMS_CACHE = Path("data/d1_teams.txt")

_ANCHOR_RE = re.compile(r"</?a\b[^>]*>")

_LINE_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})\s+"
    r"(?P<at1>@)?(?P<team1>.+?)\s+(?P<score1>\d+)\s+"
    r"(?P<at2>@)?(?P<team2>.+?)\s+(?P<score2>\d+)"
    r"(?:\s+(?P<loc>\S.*?))?\s*$"
)


def _parse_pre_block(text: str) -> pd.DataFrame:
    start = text.find("<pre>")
    end = text.find("</pre>", start)
    if start == -1 or end == -1:
        raise ValueError("Could not find <pre>...</pre> block in Massey response")
    block = text[start + len("<pre>") : end]

    rows = []
    for raw in block.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        team1 = m["team1"].strip()
        team2 = m["team2"].strip()
        s1 = int(m["score1"])
        s2 = int(m["score2"])
        home1 = m["at1"] == "@"
        home2 = m["at2"] == "@"
        neutral = not (home1 or home2)

        if home1:
            home, away, hs, as_ = team1, team2, s1, s2
        elif home2:
            home, away, hs, as_ = team2, team1, s2, s1
        else:
            # neutral — keep team1 as "home" slot purely for storage; flag neutral
            home, away, hs, as_ = team1, team2, s1, s2

        rows.append(
            {
                "date": m["date"],
                "home_team": home,
                "away_team": away,
                "home_score": hs,
                "away_score": as_,
                "neutral": neutral,
                "location": (m["loc"] or "").strip(),
            }
        )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_raw(
    url: str = DEFAULT_URL,
    cache_path: str | Path = DEFAULT_CACHE,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch Massey scores, parse, and cache to CSV.

    Massey is behind Cloudflare, so we use curl_cffi with Chrome impersonation.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path, parse_dates=["date"])
        return df

    from curl_cffi import requests as cffi_requests

    resp = cffi_requests.get(url, impersonate="chrome", timeout=30)
    resp.raise_for_status()
    df = _parse_pre_block(resp.text)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def filter_to_core(games: pd.DataFrame, min_games: int = 15) -> pd.DataFrame:
    """Iteratively prune teams with fewer than ``min_games`` against the surviving set.

    Massey's ``all=1`` page includes occasional games against non-D1 opponents
    (D2 / D3 / NAIA), and those teams typically appear only 1–6 times in the
    dataset while real D1 teams play 30+ games. Pruning under-connected teams
    drops the noise without needing an external D1 roster.
    """
    teams = set(games["home_team"]) | set(games["away_team"])
    while True:
        df = games[games["home_team"].isin(teams) & games["away_team"].isin(teams)]
        counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
        keep = set(counts[counts >= min_games].index)
        if keep == teams:
            return df.reset_index(drop=True)
        teams = keep


def fetch_d1_teams(
    url: str = D1_ONLY_URL,
    cache_path: str | Path = D1_TEAMS_CACHE,
    refresh: bool = False,
) -> set[str]:
    """Return the set of D1 team names by parsing Massey's D1-only scores page.

    The D1-only page (no &all=1) only contains games between two D1 teams, so
    the union of team names appearing on it is exactly the D1 universe.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        return {line.strip() for line in cache_path.read_text().splitlines() if line.strip()}

    from curl_cffi import requests as cffi_requests

    resp = cffi_requests.get(url, impersonate="chrome", timeout=30)
    resp.raise_for_status()
    # Strip <a href=...>name</a> wrappers so the regular parser sees plain text.
    stripped = _ANCHOR_RE.sub("", resp.text)
    df = _parse_pre_block(stripped)
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("\n".join(teams) + "\n")
    return set(teams)
