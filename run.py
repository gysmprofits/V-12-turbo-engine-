from __future__ import annotations

import os
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

import math
import pandas as pd
import yaml
import requests


# ----------------------------
# Utilities
# ----------------------------
def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def make_run_id() -> str:
    return utcnow().strftime("%Y%m%d") + "-" + uuid.uuid4().hex[:10]

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)

def write_text(path: str, s: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)


# ----------------------------
# NHL Adapter (public endpoints; tries new + legacy)
# ----------------------------
class NHLAdapter:
    name = "nhl_public_api"

    def _get(self, url: str, params: Optional[dict] = None) -> dict:
        r = requests.get(
            url,
            params=params,
            timeout=25,
            headers={"User-Agent": "matchup-engine/1.0"},
        )
        r.raise_for_status()
        return r.json()

    def _run_date(self, cfg: Dict[str, Any]) -> str:
        # YYYY-MM-DD, blank = today UTC
        d = ((cfg.get("run") or {}).get("run_date") or "").strip()
        return d if d else utcnow().date().isoformat()

    def ingest(self, cfg: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        date = self._run_date(cfg)
        notes: List[str] = []

        # ---- Schedule ----
        schedule_df = pd.DataFrame(columns=["match_id", "league", "start_time_utc", "home", "away", "format"])

        # Try NEW schedule endpoint
        try:
            js = self._get(f"https://api-web.nhle.com/v1/schedule/{date}")
            games: List[dict] = []
            blocks = []
            if isinstance(js, dict) and isinstance(js.get("gameWeek"), list):
                blocks = js["gameWeek"]
            elif isinstance(js, dict) and isinstance(js.get("games"), list):
                blocks = [{"games": js["games"]}]

            for b in blocks:
                for g in (b.get("games") or []):
                    match_id = str(g.get("id") or g.get("gameId") or g.get("gamePk") or "")
                    home = (g.get("homeTeam") or {}).get("abbrev") or (g.get("homeTeam") or {}).get("name")
                    away = (g.get("awayTeam") or {}).get("abbrev") or (g.get("awayTeam") or {}).get("name")
                    start = g.get("startTimeUTC") or g.get("startTime") or g.get("gameDate")
                    if match_id and home and away and start:
                        games.append(
                            {
                                "match_id": match_id,
                                "league": cfg["run"]["league"],
                                "start_time_utc": pd.to_datetime(start, utc=True),
                                "home": str(home),
                                "away": str(away),
                                "format": "BO1",
                            }
                        )
            if games:
                schedule_df = pd.DataFrame(games)
        except Exception as e:
            notes.append(f"new_schedule_failed:{type(e).__name__}")

        # Try LEGACY schedule if new returned nothing
        if schedule_df.empty:
            try:
                js = self._get("https://statsapi.web.nhl.com/api/v1/schedule", params={"date": date})
                games = []
                for d in js.get("dates", []):
                    for g in d.get("games", []):
                        match_id = str(g.get("gamePk", ""))
                        start = g.get("gameDate")
                        home = ((g.get("teams", {}).get("home", {}) or {}).get("team", {}) or {}).get("name")
                        away = ((g.get("teams", {}).get("away", {}) or {}).get("team", {}) or {}).get("name")
                        if match_id and home and away and start:
                            games.append(
                                {
                                    "match_id": match_id,
                                    "league": cfg["run"]["league"],
                                    "start_time_utc": pd.to_datetime(start, utc=True),
                                    "home": str(home),
                                    "away": str(away),
                                    "format": "BO1",
                                }
                            )
                if games:
                    schedule_df = pd.DataFrame(games)
            except Exception as e:
                notes.append(f"legacy_schedule_failed:{type(e).__name__}")

        # ---- Standings -> Ratings ----
        standings = pd.DataFrame(columns=["team", "points_pct", "gf", "ga", "gp"])

        # Try NEW standings endpoint
        try:
            js = self._get(f"https://api-web.nhle.com/v1/standings/{date}")
            rows = []
            for rec in (js.get("standings", []) if isinstance(js, dict) else []):
                team = rec.get("teamAbbrev") or rec.get("teamName") or rec.get("teamCommonName") or ""
                pts_pct = rec.get("pointsPct") or rec.get("pointPct") or None
                gf = rec.get("goalFor") or rec.get("goalsFor") or None
                ga = rec.get("goalAgainst") or rec.get("goalsAgainst") or None
                gp = rec.get("gamesPlayed") or None
                if team:
                    rows.append({"team": str(team), "points_pct": pts_pct, "gf": gf, "ga": ga, "gp": gp})
            if rows:
                standings = pd.DataFrame(rows)
        except Exception as e:
            notes.append(f"new_standings_failed:{type(e).__name__}")

        # Try LEGACY standings if new returned nothing
        if standings.empty:
            try:
                js = self._get("https://statsapi.web.nhl.com/api/v1/standings", params={"date": date})
                rows = []
                for rec in js.get("records", []):
                    for tr in rec.get("teamRecords", []):
                        team = (tr.get("team", {}) or {}).get("name")
                        gp = tr.get("gamesPlayed")
                        points = tr.get("points")
                        pts_pct = (points / (2 * gp)) if gp and points is not None else None
                        gf = tr.get("goalsScored")
                        ga = tr.get("goalsAgainst")
                        if team:
                            rows.append({"team": str(team), "points_pct": pts_pct, "gf": gf, "ga": ga, "gp": gp})
                if rows:
                    standings = pd.DataFrame(rows)
            except Exception as e:
                notes.append(f"legacy_standings_failed:{type(e).__name__}")

        # Numeric cleanup
        for c in ["points_pct", "gf", "ga", "gp"]:
            if c in standings.columns:
                standings[c] = pd.to_numeric(standings[c], errors="coerce")

        standings["gp"] = standings.get("gp", pd.Series(dtype=float)).fillna(0)
        standings["points_pct"] = standings.get("points_pct", pd.Series(dtype=float)).fillna(0.5)
        gf = standings.get("gf", pd.Series([0] * len(standings), dtype=float)).fillna(0.0)
        ga = standings.get("ga", pd.Series([0] * len(standings), dtype=float)).fillna(0.0)
        gp_safe = standings["gp"].replace({0: pd.NA})

        standings["gd_per_game"] = ((gf - ga) / gp_safe).fillna(0.0)

        # Elo-ish rating baseline
        standings["rating"] = 1500 + 800 * (standings["points_pct"] - 0.5) + 120 * standings["gd_per_game"]

        ratings = standings[["team", "rating"]].copy()

        # Availability placeholder (we can add injuries/goalie later)
        availability = ratings[["team"]].copy()
        availability["avail"] = 1.0

        return {
            "schedule": schedule_df,
            "ratings": ratings,
            "availability": availability,
            "notes": pd.DataFrame({"note": notes}),
        }

    def normalize(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        return tables

    def build_features(self, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        sched = tables["schedule"].copy()
        ratings = tables["ratings"].set_index("team") if not tables["ratings"].empty else pd.DataFrame().set_index("team")
        avail = tables["availability"].set_index("team") if not tables["availability"].empty else pd.DataFrame().set_index("team")

        rows = []
        for _, m in sched.iterrows():
            home, away = m["home"], m["away"]

            r_home = float(ratings.loc[home, "rating"]) if home in ratings.index else 1500.0
            r_away = float(ratings.loc[away, "rating"]) if away in ratings.index else 1500.0
            a_home = float(avail.loc[home, "avail"]) if home in avail.index else 1.0
            a_away = float(avail.loc[away, "avail"]) if away in avail.index else 1.0

            rows.append(
                {
                    "match_id": m["match_id"],
                    "league": m["league"],
                    "start_time_utc": m["start_time_utc"],
                    "home": home,
                    "away": away,
                    "format": m.get("format", "BO1"),
                    "rating_diff": r_home - r_away,
                    "avail_diff": a_home - a_away,
                    "home_adv": 45.0,  # home-ice bump (tune later)
                }
            )
        return pd.DataFrame(rows)

    def score(self, feats: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
        out = []
        for _, r in feats.iterrows():
            d = float(r["rating_diff"]) + float(r.get("home_adv", 0.0))
            # Elo win prob
            p_home = 1.0 / (1.0 + (10 ** (-d / 400.0)))
            p_away = 1.0 - p_home

            edge = abs(p_home - 0.5) * 2.0
            uncertainty = 1.0 - edge
            confidence = edge

            min_conf = float(cfg["risk"]["min_confidence"])
            max_unc = float(cfg["risk"]["max_uncertainty"])
            recommended = (confidence >= min_conf) and (uncertainty <= max_unc)

            reasons = ["HOME_ICE"]
            if abs(float(r["rating_diff"])) >= 40:
                reasons.append("STANDINGS_RATING_EDGE")

            out.append(
                {
                    "match_id": r["match_id"],
                    "home": r["home"],
                    "away": r["away"],
                    "format": r.get("format", "BO1"),
                    "p_home": round(p_home, 4),
                    "p_away": round(p_away, 4),
                    "edge": round(edge, 4),
                    "confidence": round(confidence, 4),
                    "uncertainty": round(uncertainty, 4),
                    "recommended": bool(recommended),
                    "reason_codes": ",".join(reasons),
                }
            )
        return pd.DataFrame(out)


# ----------------------------
# Engine (fully sequential)
# ----------------------------
def main() -> None:
    cfg_path = os.environ.get("ENGINE_CONFIG", "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rid = make_run_id()
    root = cfg["paths"]["data_root"]
    out_dir = os.path.join(root, "exports", rid)
    ensure_dir(out_dir)

    manifest = {
        "run_id": rid,
        "started_at": utcnow().isoformat(),
        "league": cfg["run"]["league"],
        "slate_id": cfg["run"]["slate_id"],
        "adapter": "nhl_public_api",
        "degraded_mode": False,
        "notes": [],
    }

    adapter = NHLAdapter()

    # V2 Ingest
    tables = adapter.ingest(cfg)

    # V3 Normalize
    tables = adapter.normalize(tables)

    # V4 Quality gate
    sched = tables["schedule"]
    if len(sched) < int(cfg["quality_gates"]["min_matchups"]):
        manifest["degraded_mode"] = True
        manifest["notes"].append("QUALITY_FAIL: schedule too small (no games or API mismatch)")

    # V5 Features
    feats = adapter.build_features(tables)

    # V7 Score
    scores = adapter.score(feats, cfg) if not feats.empty else pd.DataFrame(
        columns=["match_id","home","away","format","p_home","p_away","edge","confidence","uncertainty","recommended","reason_codes"]
    )

    # V9 Write-ups
    cards = []
    for _, s in scores.iterrows():
        pick_side = s["home"] if float(s["p_home"]) >= 0.5 else s["away"]
        p_pick = s["p_home"] if float(s["p_home"]) >= 0.5 else s["p_away"]
        status = "✅ RECOMMENDED" if bool(s["recommended"]) else "⚠️ NO-PLAY"
        cards.append(
            f"### {s['home']} vs {s['away']} ({s['format']})\n"
            f"- Pick: **{pick_side}** (p={p_pick})\n"
            f"- Confidence: **{s['confidence']}** | Uncertainty: {s['uncertainty']} | Edge: {s['edge']}\n"
            f"- Status: {status}\n"
            f"- Reason codes: {s['reason_codes'] or '—'}\n"
        )

    slate = []
    slate.append(f"# Slate Summary — {cfg['run']['league']} — {manifest['started_at']}\n")
    if scores.empty:
        slate.append("No games found for the run date, or API parsing didn’t return games.\n")
    else:
        best = scores.sort_values(["recommended", "confidence", "edge"], ascending=[False, False, False])
        slate.append("## Top edges\n")
        for _, s in best.head(10).iterrows():
            side = "HOME" if float(s["p_home"]) >= 0.5 else "AWAY"
            slate.append(f"- {s['home']} vs {s['away']}: pick={side} (conf={s['confidence']}, edge={s['edge']}) "
                         f"{'✅' if bool(s['recommended']) else '⚠️'}")
        slate.append("\n## No-play list\n")
        for _, s in best[best["recommended"] == False].iterrows():
            slate.append(f"- {s['home']} vs {s['away']} (uncert={s['uncertainty']}, conf={s['confidence']})")

    # V11 Exports
    write_json(os.path.join(out_dir, "run_manifest.json"), manifest)
    feats.to_parquet(os.path.join(out_dir, "features.parquet"), index=False)
    scores.to_csv(os.path.join(out_dir, "scores.csv"), index=False)
    write_json(os.path.join(out_dir, "scores.json"), scores.to_dict(orient="records"))
    write_text(os.path.join(out_dir, "matchup_cards.md"), "\n\n".join(cards) if cards else "No matchup cards generated.\n")
    write_text(os.path.join(out_dir, "slate_summary.md"), "\n".join(slate))

    print(f"RUN COMPLETE: {rid}")
    print(f"EXPORTS: {out_dir}")


if __name__ == "__main__":
    main()
