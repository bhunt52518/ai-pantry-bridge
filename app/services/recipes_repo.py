# app/services/recipes_repo.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from app.services import common


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_recipe_parsed(
    *,
    title: str,
    source_url: Optional[str],
    parsed: dict[str, Any],
    raw_text: Optional[str] = None,
) -> int:
    """
    Saves into the `recipes` table (id INTEGER AUTOINCREMENT).
    Stores the normalized parsed JSON in parsed_json.
    """
    now = _now_iso()
    parsed_json = json.dumps(parsed, ensure_ascii=False)

    with common.pantry_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO recipes (title, source_url, raw_text, parsed_json, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (title.strip(), source_url, raw_text, parsed_json, now),
        )
        conn.commit()
        return int(cur.lastrowid)


def save_recipe_raw(
    *,
    recipe_id: str,
    title: str,
    source_url: Optional[str],
    payload_json: str,
) -> None:
    """
    Saves into `recipes_raw` (id TEXT PK). This is your "raw archive".
    """
    now = _now_iso()
    with common.pantry_db() as conn:
        conn.execute(
            """
            INSERT INTO recipes_raw (id, created_at, title, source_url, json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (recipe_id, now, title.strip(), source_url, payload_json),
        )
        conn.commit()


def list_recipes(limit: int = 50) -> list[dict[str, Any]]:
    with common.pantry_db() as conn:
        rows = conn.execute(
            """
            SELECT id, title, source_url, updated_at
            FROM recipes
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()

    return [
        {"id": r[0], "title": r[1], "source_url": r[2], "updated_at": r[3]}
        for r in rows
    ]


def get_recipe(recipe_id: int) -> Optional[dict[str, Any]]:
    with common.pantry_db() as conn:
        row = conn.execute(
            """
            SELECT id, title, source_url, raw_text, parsed_json, updated_at
            FROM recipes
            WHERE id = ?
            """,
            (int(recipe_id),),
        ).fetchone()

    if not row:
        return None

    parsed_json = row[4]
    try:
        parsed = json.loads(parsed_json) if parsed_json else None
    except Exception:
        parsed = None

    return {
        "id": row[0],
        "title": row[1],
        "source_url": row[2],
        "raw_text": row[3],
        "parsed": parsed,
        "updated_at": row[5],
    }
