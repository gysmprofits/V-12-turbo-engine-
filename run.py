from __future__ import annotations
import os, json, uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any
import pandas as pd
import yaml

# ----------------------------
# Utilities
# ----------------------------
def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def run_id() -> str:
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
# Demo "universal adapter"
# Replace this later with real sport/esport adapters
# ----------------------------
class DemoAdapter:
    name = "demo_universal"

    def ingest(self, cfg: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        # Minimal schedule + basic team strength inputs (demo)
        now = utcnow()
        schedule = pd.DataFrame([
            {"match_id": "M1", "league": cfg["run"]["league"], "start_time_utc": now, "home": "TEAM_A", "away": "TEAM_B", "format": "BO1"},
            {"match_id": "M2", "league": cfg["run"]["league"], "start_time_utc": now, "home": "TEAM_C", "away": "TEAM_D", "format": "BO3"},
        ])
        # Pretend these are learned ratings or power scores
        ratings = pd.DataFrame([
            {"team": "TEAM_A", "rating": 1600},
            {"team": "TEAM_B", "rating": 1500},
            {"team": "TEAM_C", "rating": 1450},
            {"team": "TEAM_D", "rating": 1550},
        ])
        # Availability/injury-like signal (0=bad, 1=good)
        availability = pd.DataFrame([
            {"team": "TEAM_A", "avail": 0.90},
            {"team": "TEAM_B", "avail": 0.95},
            {"team": "TEAM_C", "avail": 0.70},  # more risk
            {"team": "TEAM_D", "avail": 0.85},
        ])
        return {"schedule": schedule, "ratings": ratings, "availability": availability}

    def normalize(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        # In real life: entity resolution, timestamp alignment, ID mapping, etc.
        return tables

    def build_features(self, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        sched = tables["schedule"]
        ratings = tables["ratings"].set_index("team")
        avail = tables["availability"].set_index("team")

        rows = []
        for _, m in sched.iterrows():
            home, away = m["home"], m["away"]
            r_home = float(ratings.loc[home, "rating"])
            r_away = float(ratings.loc[away, "rating"])
            a_home = float(avail.loc[home, "avail"])
            a_away = float(avail.loc[away, "avail"])
            rows.append({
                "match_id": m["match_id"],
                "league": m["league"],
                "start_time_utc": m["start_time_utc"],
                "home": home,
                "away": away,
                "format": m["format"],
                "rating_diff": r_home - r_away,
                "avail_diff": a_home - a_away,
                "home_avail": a_home,
                "away_avail": a_away,
            })
        return pd.DataFrame(rows)

    def score(self, feats: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
        # Fast baseline scoring: logistic on rating/availability diffs (demo)
        # This is intentionally simple; you’ll replace with your real model head.
        import math

        def sigmoid(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-x))

        # scale factors (tunable)
        k_rating = 1 / 400.0
        k_avail = 1.5

        out = []
        for _, r in feats.iterrows():
            x = (r["rating_diff"] * k_rating) + (r["avail_diff"] * k_avail)
            p_home = sigmoid(x)
            p_away = 1.0 - p_home

            edge = abs(p_home - 0.5) * 2.0  # 0..1
            uncertainty = 1.0 - edge       # crude proxy
            confidence = 1.0 - uncertainty

            recommended = (confidence >= cfg["risk"]["min_confidence"]) and (uncertainty <= cfg["risk"]["max_uncertainty"])

            reasons = []
            if abs(r["rating_diff"]) >= 80:
                reasons.append("RATING_EDGE")
            if abs(r["avail_diff"]) >= 0.10:
                reasons.append("AVAILABILITY_EDGE")
            if r["format"] != "BO1":
                reasons.append("SERIES_FORMAT")

            out.append({
                "match_id": r["match_id"],
                "home": r["home"],
                "away": r["away"],
                "format": r["format"],
                "p_home": round(p_home, 4),
                "p_away": round(p_away, 4),
                "edge": round(edge, 4),
                "confidence": round(confidence, 4),
                "uncertainty": round(uncertainty, 4),
                "recommended": bool(recommended),
                "reason_codes": ",".join(reasons),
            })
        return pd.DataFrame(out)

# ----------------------------
# Engine (fully sequential blocks)
# ----------------------------
def main():
    cfg_path = os.environ.get("ENGINE_CONFIG", "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    RID = run_id()
    root = cfg["paths"]["data_root"]
    out_dir = os.path.join(root, "exports", RID)
    ensure_dir(out_dir)

    manifest = {
        "run_id": RID,
        "started_at": utcnow().isoformat(),
        "league": cfg["run"]["league"],
        "slate_id": cfg["run"]["slate_id"],
        "adapter": "demo_universal",
        "degraded_mode": False,
        "notes": [],
    }

    adapter = DemoAdapter()

    # V2 Ingest
    tables = adapter.ingest(cfg)

    # V3 Normalize
    tables = adapter.normalize(tables)

    # V4 Quality gates
    sched = tables["schedule"]
    if len(sched) < cfg["quality_gates"]["min_matchups"]:
        manifest["notes"].append("QUALITY_FAIL: schedule too small")
        manifest["degraded_mode"] = True

    # V5 Features
    feats = adapter.build_features(tables)

    # V7 Score
    scores = adapter.score(feats, cfg)

    # V9 Write-ups
    cards = []
    for _, s in scores.iterrows():
        pick_side = s["home"] if s["p_home"] >= 0.5 else s["away"]
        p_pick = s["p_home"] if s["p_home"] >= 0.5 else s["p_away"]
        status = "✅ RECOMMENDED" if s["recommended"] else "⚠️ NO-PLAY"
        cards.append(
            f"### {s['home']} vs {s['away']} ({s['format']})\n"
            f"- Pick: **{pick_side}** (p={p_pick})\n"
            f"- Confidence: **{s['confidence']}** | Uncertainty: {s['uncertainty']} | Edge: {s['edge']}\n"
            f"- Status: {status}\n"
            f"- Reason codes: {s['reason_codes'] or '—'}\n"
        )

    slate_summary = []
    best = scores.sort_values(["recommended", "confidence", "edge"], ascending=[False, False, False])
    slate_summary.append(f"# Slate Summary — {cfg['run']['league']} — {manifest['started_at']}\n")
    slate_summary.append("## Top edges\n")
    for _, s in best.head(10).iterrows():
        slate_summary.append(f"- {s['home']} vs {s['away']}: pick={'HOME' if s['p_home']>=0.5 else 'AWAY'} "
                             f"(conf={s['confidence']}, edge={s['edge']}) {'✅' if s['recommended'] else '⚠️'}")
    slate_summary.append("\n## No-play list\n")
    for _, s in best[best["recommended"] == False].iterrows():
        slate_summary.append(f"- {s['home']} vs {s['away']} (uncert={s['uncertainty']}, conf={s['confidence']})")

    # V11 Exports
    write_json(os.path.join(out_dir, "run_manifest.json"), manifest)
    feats.to_parquet(os.path.join(out_dir, "features.parquet"), index=False)
    scores.to_csv(os.path.join(out_dir, "scores.csv"), index=False)
    write_json(os.path.join(out_dir, "scores.json"), scores.to_dict(orient="records"))
    write_text(os.path.join(out_dir, "matchup_cards.md"), "\n\n".join(cards))
    write_text(os.path.join(out_dir, "slate_summary.md"), "\n".join(slate_summary))

    print(f"RUN COMPLETE: {RID}")
    print(f"EXPORTS: {out_dir}")

if __name__ == "__main__":
    main()
