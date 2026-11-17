"""SQLite persistence for analysis records and dashboard aggregates."""
from __future__ import annotations

import datetime as dt
import json
import sqlite3

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    text                  TEXT    NOT NULL,
    source                TEXT,
    platform              TEXT,
    modality              TEXT    NOT NULL,
    image_name            TEXT,
    image_url             TEXT,
    image_sha256          TEXT,
    image_width           INTEGER,
    image_height          INTEGER,
    label                 TEXT    NOT NULL,
    threshold             REAL    DEFAULT 0.5,
    misleading_prob       REAL    NOT NULL,
    confidence            REAL    NOT NULL,
    text_score            REAL    NOT NULL,
    image_score           REAL,
    consistency_score     REAL,
    consistency_label     TEXT,
    consistency_cosine    REAL,
    contribution_text     REAL,
    contribution_image    REAL,
    text_json             TEXT,
    image_json            TEXT,
    consistency_json      TEXT,
    explanation           TEXT,
    latency_ms            INTEGER,
    created_at            TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analyses_label   ON analyses (label);
CREATE INDEX IF NOT EXISTS idx_analyses_modality ON analyses (modality);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DATABASE_PATH, timeout=20)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def insert(result: dict) -> int:
    text_analysis = result.get("text_analysis") or {}
    image_analysis = result.get("image_analysis") or {}
    stats = (image_analysis.get("statistics") or {})
    consistency = result.get("consistency") or {}

    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO analyses (
                text, source, platform, modality, image_name, image_url, image_sha256,
                image_width, image_height, label, threshold, misleading_prob, confidence,
                text_score, image_score, consistency_score, consistency_label,
                consistency_cosine, contribution_text, contribution_image,
                text_json, image_json, consistency_json, explanation, latency_ms, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                result["text"], result.get("source"), result.get("platform"),
                result["modality"], stats.get("filename"), stats.get("url"),
                stats.get("sha256"), stats.get("width"), stats.get("height"),
                result["label"], result.get("threshold", 0.5),
                result["misleading_prob"], result["confidence"],
                text_analysis.get("score"), (image_analysis or {}).get("score"),
                consistency.get("score"), consistency.get("label"),
                consistency.get("cosine_similarity"),
                result["contributions"].get("text"),
                result["contributions"].get("image"),
                json.dumps(text_analysis), json.dumps(image_analysis or {}),
                json.dumps(consistency or {}), result["explanation"],
                result.get("latency_ms"),
                dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            ),
        )
        return int(cur.lastrowid)


def count() -> int:
    with connect() as conn:
        return int(conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0])


def _row_to_result(row: sqlite3.Row) -> dict:
    data = dict(row)
    data["text_analysis"] = json.loads(data.pop("text_json") or "{}")
    data["image_analysis"] = json.loads(data.pop("image_json") or "{}")
    data["consistency"] = json.loads(data.pop("consistency_json") or "{}") or None
    data["contributions"] = {
        "text": data.get("contribution_text"),
        "image": data.get("contribution_image"),
        "method": "stored at analysis time",
    }
    data["stored"] = True
    return data


def get(analysis_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    return _row_to_result(row) if row else None


def recent(limit: int = 8) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, text, label, confidence, modality, platform, source,"
            " consistency_label, created_at FROM analyses"
            " ORDER BY id DESC LIMIT ?", (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def list_page(query: str = "", label: str = "", modality: str = "",
              page: int = 1, per_page: int = 10) -> dict:
    where, params = [], []
    if query:
        where.append("(text LIKE ? OR source LIKE ? OR platform LIKE ?)")
        like = f"%{query}%"
        params.extend([like, like, like])
    if label in ("Credible", "Misleading"):
        where.append("label = ?")
        params.append(label)
    if modality in ("text", "multimodal"):
        where.append("modality = ?")
        params.append(modality)
    clause = (" WHERE " + " AND ".join(where)) if where else ""

    with connect() as conn:
        total = int(conn.execute(
            f"SELECT COUNT(*) FROM analyses{clause}", params).fetchone()[0])
        pages = max(1, (total + per_page - 1) // per_page)
        page = max(1, min(page, pages))
        rows = conn.execute(
            f"SELECT id, text, label, confidence, modality, platform, source,"
            f" consistency_label, consistency_score, created_at FROM analyses{clause}"
            f" ORDER BY id DESC LIMIT ? OFFSET ?",
            (*params, per_page, (page - 1) * per_page),
        ).fetchall()

    return {
        "rows": [dict(r) for r in rows],
        "total": total,
        "page": page,
        "pages": pages,
        "per_page": per_page,
        "query": query,
        "label": label,
        "modality": modality,
    }


def stats() -> dict:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS total,
                   SUM(label = 'Credible')   AS credible,
                   SUM(label = 'Misleading') AS misleading,
                   AVG(confidence)           AS avg_confidence,
                   AVG(misleading_prob)      AS avg_misleading_prob,
                   SUM(modality = 'text')    AS text_only,
                   SUM(modality = 'multimodal') AS multimodal,
                   SUM(consistency_label = 'Inconsistent') AS inconsistent,
                   AVG(latency_ms)           AS avg_latency,
                   MAX(created_at)           AS last_seen
            FROM analyses
            """
        ).fetchone()
        week_ago = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=7)).isoformat(
            timespec="seconds")
        last_week = int(conn.execute(
            "SELECT COUNT(*) FROM analyses WHERE created_at >= ?", (week_ago,)
        ).fetchone()[0])

    total = int(row["total"] or 0)
    credible = int(row["credible"] or 0)
    misleading = int(row["misleading"] or 0)
    return {
        "total": total,
        "credible": credible,
        "misleading": misleading,
        "credible_pct": round(100 * credible / total, 1) if total else 0.0,
        "misleading_pct": round(100 * misleading / total, 1) if total else 0.0,
        "avg_confidence": round(float(row["avg_confidence"] or 0) * 100, 1),
        "avg_misleading_prob": round(float(row["avg_misleading_prob"] or 0) * 100, 1),
        "text_only": int(row["text_only"] or 0),
        "multimodal": int(row["multimodal"] or 0),
        "inconsistent": int(row["inconsistent"] or 0),
        "avg_latency_ms": int(row["avg_latency"] or 0),
        "last_seen": row["last_seen"],
        "last_7_days": last_week,
    }


def trend(days: int = 14) -> dict:
    today = dt.datetime.now(dt.timezone.utc).date()
    start = today - dt.timedelta(days=days - 1)
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT substr(created_at, 1, 10) AS day,
                   SUM(label = 'Credible')   AS credible,
                   SUM(label = 'Misleading') AS misleading,
                   AVG(confidence)           AS confidence
            FROM analyses
            WHERE created_at >= ?
            GROUP BY day ORDER BY day
            """,
            (start.isoformat(),),
        ).fetchall()
    by_day = {r["day"]: r for r in rows}
    labels, credible, misleading, confidence = [], [], [], []
    for offset in range(days):
        day = (start + dt.timedelta(days=offset)).isoformat()
        labels.append(day[5:])
        entry = by_day.get(day)
        credible.append(int(entry["credible"] or 0) if entry else 0)
        misleading.append(int(entry["misleading"] or 0) if entry else 0)
        confidence.append(round(float(entry["confidence"] or 0) * 100, 1) if entry else 0)
    return {"labels": labels, "credible": credible, "misleading": misleading,
            "confidence": confidence}


def platform_breakdown() -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT COALESCE(NULLIF(platform, ''), 'Unspecified') AS platform,
                   COUNT(*) AS total,
                   SUM(label = 'Misleading') AS misleading
            FROM analyses GROUP BY platform ORDER BY total DESC LIMIT 6
            """
        ).fetchall()
    return [dict(r) for r in rows]


def delete(analysis_id: int) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
